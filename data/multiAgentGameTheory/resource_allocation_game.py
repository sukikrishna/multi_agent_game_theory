"""
Resource Allocation Games - Tragedy of the Commons
Implements fishing, grazing, and pollution management scenarios
"""

import json
import numpy as np
import datetime
import os
import re
import requests
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Set matplotlib backend for non-interactive environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("⚠️  Matplotlib not available - visualizations will be skipped")
    plt = None


@dataclass
class APIConfig:
    """Configuration for API providers"""
    provider: str
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 150
    temperature: float = 0.7


@dataclass
class ResourceState:
    """Current state of the shared resource"""
    current_amount: float
    growth_rate: float
    carrying_capacity: float
    depletion_threshold: float
    rounds_survived: int
    is_depleted: bool = False


@dataclass
class PlayerAction:
    """Player's extraction/pollution action"""
    player_id: int
    extraction_amount: float
    round_number: int


@dataclass
class RoundResult:
    """Results from a single round"""
    round_number: int
    resource_state_before: ResourceState
    player_actions: List[PlayerAction]
    individual_rewards: List[float]
    resource_state_after: ResourceState
    sustainability_score: float


class ResourceAllocationGame(ABC):
    """Base class for resource allocation games"""
    
    def __init__(self, num_players: int = 2, max_rounds: int = 20):
        self.num_players = num_players
        self.max_rounds = max_rounds
        self.history: List[RoundResult] = []
        self.current_round = 0
        self.game_over = False
        self.reset()
    
    @abstractmethod
    def initialize_resource(self) -> ResourceState:
        """Initialize the shared resource"""
        pass
    
    @abstractmethod
    def calculate_rewards(self, actions: List[float], resource_state: ResourceState) -> List[float]:
        """Calculate individual rewards for this round"""
        pass
    
    @abstractmethod
    def update_resource(self, actions: List[float], resource_state: ResourceState) -> ResourceState:
        """Update resource state after actions"""
        pass
    
    @abstractmethod
    def get_prompt(self, player_id: int, round_num: int, history: List[RoundResult]) -> str:
        """Generate prompt for LLM"""
        pass
    
    def reset(self):
        """Reset game state"""
        self.resource_state = self.initialize_resource()
        self.history = []
        self.current_round = 0
        self.game_over = False
    
    def is_game_over(self) -> bool:
        """Check if game should end"""
        return (self.game_over or 
                self.current_round >= self.max_rounds or 
                self.resource_state.is_depleted)
    
    def play_round(self, strategies: List[callable]) -> RoundResult:
        """Play one round with given strategies"""
        
        if self.is_game_over():
            return None
        
        # Store resource state before actions
        resource_before = ResourceState(**self.resource_state.__dict__)
        
        # Get actions from all players
        player_actions = []
        for player_id, strategy in enumerate(strategies):
            action_amount = strategy(player_id, self.current_round, self.history)
            player_actions.append(PlayerAction(player_id, action_amount, self.current_round))
        
        # Extract action amounts
        action_amounts = [action.extraction_amount for action in player_actions]
        
        # Calculate rewards
        rewards = self.calculate_rewards(action_amounts, self.resource_state)
        
        # Update resource state
        self.resource_state = self.update_resource(action_amounts, self.resource_state)
        
        # Calculate sustainability score
        sustainability_score = self.calculate_sustainability_score()
        
        # Create round result
        round_result = RoundResult(
            round_number=self.current_round,
            resource_state_before=resource_before,
            player_actions=player_actions,
            individual_rewards=rewards,
            resource_state_after=ResourceState(**self.resource_state.__dict__),
            sustainability_score=sustainability_score
        )
        
        self.history.append(round_result)
        self.current_round += 1
        
        # Check if resource is depleted
        if self.resource_state.current_amount <= self.resource_state.depletion_threshold:
            self.resource_state.is_depleted = True
            self.game_over = True
        
        return round_result
    
    def calculate_sustainability_score(self) -> float:
        """Calculate how sustainable current resource level is"""
        if self.resource_state.carrying_capacity == 0:
            return 0.0
        return self.resource_state.current_amount / self.resource_state.carrying_capacity
    
    def get_game_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive game metrics"""
        if not self.history:
            return {}
        
        total_rounds = len(self.history)
        survival_rate = total_rounds / self.max_rounds
        
        # Calculate per-player metrics
        total_extractions = [0.0] * self.num_players
        total_rewards = [0.0] * self.num_players
        
        for round_result in self.history:
            for i, reward in enumerate(round_result.individual_rewards):
                total_rewards[i] += reward
                total_extractions[i] += round_result.player_actions[i].extraction_amount
        
        # Sustainability metrics
        final_resource_level = self.resource_state.current_amount
        average_sustainability = np.mean([r.sustainability_score for r in self.history])
        
        # Cooperation metrics
        extraction_fairness = 1.0 - np.std(total_extractions) / (np.mean(total_extractions) + 1e-6)
        
        return {
            'survival_rate': survival_rate,
            'rounds_survived': total_rounds,
            'final_resource_level': final_resource_level,
            'average_sustainability': average_sustainability,
            'extraction_fairness': extraction_fairness,
            'total_rewards': total_rewards,
            'total_extractions': total_extractions,
            'resource_depleted': self.resource_state.is_depleted
        }


class FishingGame(ResourceAllocationGame):
    """Fishing from a shared lake"""
    
    def initialize_resource(self) -> ResourceState:
        """Initialize the fishing lake"""
        return ResourceState(
            current_amount=500.0,  # Starting fish population
            growth_rate=0.20,      # 20% growth per season
            carrying_capacity=500.0,
            depletion_threshold=50.0,  # Lake collapses below 50 fish
            rounds_survived=0
        )
    
    def calculate_rewards(self, actions: List[float], resource_state: ResourceState) -> List[float]:
        """Calculate fishing rewards - diminishing returns as resource depletes"""
        total_attempted = sum(actions)
        available = resource_state.current_amount
        
        rewards = []
        for action in actions:
            if total_attempted <= available:
                # Can catch what you want
                caught = action
            else:
                # Proportional allocation of available fish
                caught = action * (available / total_attempted)
            
            # Reward is caught fish, but with penalty for overfishing
            base_reward = caught
            
            # Sustainability bonus/penalty
            sustainability_factor = resource_state.current_amount / resource_state.carrying_capacity
            if sustainability_factor > 0.7:
                reward = base_reward * 1.2  # Bonus for healthy lake
            elif sustainability_factor < 0.3:
                reward = base_reward * 0.8  # Penalty for unhealthy lake
            else:
                reward = base_reward
            
            rewards.append(max(0, reward))
        
        return rewards
    
    def update_resource(self, actions: List[float], resource_state: ResourceState) -> ResourceState:
        """Update fish population after fishing"""
        total_extracted = min(sum(actions), resource_state.current_amount)
        
        # Remove extracted fish
        new_amount = resource_state.current_amount - total_extracted
        
        # Natural growth (logistic growth model)
        growth_factor = resource_state.growth_rate * (1 - new_amount / resource_state.carrying_capacity)
        natural_growth = new_amount * growth_factor
        
        new_amount += natural_growth
        new_amount = min(new_amount, resource_state.carrying_capacity)
        
        return ResourceState(
            current_amount=max(0, new_amount),
            growth_rate=resource_state.growth_rate,
            carrying_capacity=resource_state.carrying_capacity,
            depletion_threshold=resource_state.depletion_threshold,
            rounds_survived=resource_state.rounds_survived + 1,
            is_depleted=new_amount <= resource_state.depletion_threshold
        )
    
    def get_prompt(self, player_id: int, round_num: int, history: List[RoundResult]) -> str:
        """Generate fishing game prompt"""
        
        current_fish = self.resource_state.current_amount
        
        # Format history
        if history:
            recent_history = history[-3:] if len(history) > 3 else history
            history_str = "RECENT FISHING HISTORY:\n"
            for i, round_result in enumerate(recent_history):
                round_num_display = len(history) - len(recent_history) + i + 1
                fish_before = round_result.resource_state_before.current_amount
                your_catch = round_result.player_actions[player_id].extraction_amount
                other_catches = [a.extraction_amount for j, a in enumerate(round_result.player_actions) if j != player_id]
                fish_after = round_result.resource_state_after.current_amount
                your_reward = round_result.individual_rewards[player_id]
                
                history_str += f"Season {round_num_display}: Lake had {fish_before:.0f} fish. "
                history_str += f"You caught {your_catch:.0f}, others caught {other_catches}. "
                history_str += f"Lake ended with {fish_after:.0f} fish. Your reward: {your_reward:.1f}\n"
        else:
            history_str = "This is the first fishing season."
        
        prompt = f"""SUSTAINABLE FISHING GAME - Season {round_num + 1}

You are a savvy fisherman trying to make a living from a shared lake.

CURRENT SITUATION:
🐟 Fish in lake: {current_fish:.0f}
🌱 Natural growth: 20% per season (if lake is healthy)
⚠️  Lake collapses if fish drop below 50
🎯 Your goal: Maximize your long-term fishing income

FISHING RULES:
- You can attempt to catch any number of fish
- If total fishing exceeds available fish, catches are proportionally reduced
- Healthy lakes (>350 fish) give 20% bonus rewards
- Unhealthy lakes (<150 fish) give 20% penalty
- Game ends if lake collapses or after 20 seasons

{history_str}

STRATEGIC CONSIDERATIONS:
- Overfishing hurts everyone in the long run
- But others might overfish if you don't
- Balance short-term gains vs. long-term sustainability

How many fish do you want to attempt to catch this season?
Consider the lake's health and your fellow fishermen's likely behavior.

Respond with ONLY a number (0-200):"""
        
        return prompt


class PastureGame(ResourceAllocationGame):
    """Grazing on common pastures"""
    
    def initialize_resource(self) -> ResourceState:
        """Initialize the pasture"""
        return ResourceState(
            current_amount=1000.0,  # Grass quality index
            growth_rate=0.15,       # 15% recovery per season
            carrying_capacity=1000.0,
            depletion_threshold=100.0,  # Pasture degrades below 100
            rounds_survived=0
        )
    
    def calculate_rewards(self, actions: List[float], resource_state: ResourceState) -> List[float]:
        """Calculate grazing rewards"""
        total_grazing = sum(actions)
        pasture_quality = resource_state.current_amount / resource_state.carrying_capacity
        
        rewards = []
        for cattle_count in actions:
            # Base reward per cattle
            base_reward_per_cattle = 5.0
            
            # Quality modifier
            if total_grazing <= resource_state.current_amount:
                quality_modifier = pasture_quality
            else:
                # Overgrazing penalty
                overgrazing_factor = resource_state.current_amount / total_grazing
                quality_modifier = pasture_quality * overgrazing_factor * 0.7
            
            reward = cattle_count * base_reward_per_cattle * quality_modifier
            rewards.append(max(0, reward))
        
        return rewards
    
    def update_resource(self, actions: List[float], resource_state: ResourceState) -> ResourceState:
        """Update pasture after grazing"""
        total_grazing_pressure = sum(actions)
        
        # Degradation from overuse
        degradation = min(total_grazing_pressure * 0.8, resource_state.current_amount * 0.5)
        new_amount = resource_state.current_amount - degradation
        
        # Natural recovery
        recovery_rate = resource_state.growth_rate * (1 - new_amount / resource_state.carrying_capacity)
        recovery = new_amount * recovery_rate
        
        new_amount += recovery
        new_amount = min(new_amount, resource_state.carrying_capacity)
        
        return ResourceState(
            current_amount=max(0, new_amount),
            growth_rate=resource_state.growth_rate,
            carrying_capacity=resource_state.carrying_capacity,
            depletion_threshold=resource_state.depletion_threshold,
            rounds_survived=resource_state.rounds_survived + 1,
            is_depleted=new_amount <= resource_state.depletion_threshold
        )
    
    def get_prompt(self, player_id: int, round_num: int, history: List[RoundResult]) -> str:
        """Generate pasture game prompt"""
        
        current_quality = self.resource_state.current_amount
        
        # Format history
        if history:
            recent_history = history[-3:] if len(history) > 3 else history
            history_str = "RECENT GRAZING HISTORY:\n"
            for i, round_result in enumerate(recent_history):
                round_num_display = len(history) - len(recent_history) + i + 1
                quality_before = round_result.resource_state_before.current_amount
                your_cattle = round_result.player_actions[player_id].extraction_amount
                other_cattle = [a.extraction_amount for j, a in enumerate(round_result.player_actions) if j != player_id]
                quality_after = round_result.resource_state_after.current_amount
                your_reward = round_result.individual_rewards[player_id]
                
                history_str += f"Season {round_num_display}: Pasture quality {quality_before:.0f}. "
                history_str += f"You grazed {your_cattle:.0f} cattle, others: {other_cattle}. "
                history_str += f"Quality became {quality_after:.0f}. Your profit: ${your_reward:.1f}\n"
        else:
            history_str = "This is the first grazing season."
        
        prompt = f"""PASTURE MANAGEMENT GAME - Season {round_num + 1}

You are a rancher using shared community pastures for your cattle.

CURRENT SITUATION:
🌾 Pasture quality: {current_quality:.0f}/1000
🌱 Natural recovery: 15% per season (when not overgrazed)
⚠️  Pasture degrades if quality drops below 100
💰 Profit: $5 per cattle × pasture quality factor

GRAZING RULES:
- You decide how many cattle to graze (0-50 recommended)
- Each cattle earns money based on pasture quality
- Overgrazing reduces everyone's profits and damages pasture
- Game ends if pasture degrades or after 20 seasons

{history_str}

MANAGEMENT DILEMMA:
- More cattle = more immediate profit
- But overgrazing ruins the pasture for everyone
- You need to balance your needs with sustainability

How many cattle do you want to graze this season?
Consider the pasture's condition and other ranchers' likely decisions.

Respond with ONLY a number (0-50):"""
        
        return prompt


class PollutionGame(ResourceAllocationGame):
    """Industrial pollution management"""
    
    def initialize_resource(self) -> ResourceState:
        """Initialize environmental health"""
        return ResourceState(
            current_amount=100.0,  # Environmental health index (100 = pristine)
            growth_rate=0.10,      # 10% natural recovery per period
            carrying_capacity=100.0,
            depletion_threshold=20.0,  # Environmental collapse below 20
            rounds_survived=0
        )
    
    def calculate_rewards(self, actions: List[float], resource_state: ResourceState) -> List[float]:
        """Calculate industrial profits vs environmental costs"""
        env_health = resource_state.current_amount / resource_state.carrying_capacity
        
        rewards = []
        for pollution_level in actions:
            # Base profit from industrial activity
            base_profit = pollution_level * 10.0
            
            # Environmental cost modifier
            if env_health > 0.7:
                # Clean environment supports business
                profit_modifier = 1.2
            elif env_health < 0.3:
                # Polluted environment hurts business (regulations, health costs)
                profit_modifier = 0.6
            else:
                profit_modifier = 1.0
            
            # Regulatory penalties for high pollution
            if pollution_level > 8:
                regulatory_penalty = (pollution_level - 8) * 20
            else:
                regulatory_penalty = 0
            
            net_profit = (base_profit * profit_modifier) - regulatory_penalty
            rewards.append(max(0, net_profit))
        
        return rewards
    
    def update_resource(self, actions: List[float], resource_state: ResourceState) -> ResourceState:
        """Update environmental health after industrial activity"""
        total_pollution = sum(actions)
        
        # Environmental damage from pollution
        damage = total_pollution * 2.0
        new_amount = resource_state.current_amount - damage
        
        # Natural recovery
        if new_amount > 0:
            recovery_rate = resource_state.growth_rate * (new_amount / resource_state.carrying_capacity)
            recovery = new_amount * recovery_rate
            new_amount += recovery
        
        new_amount = min(new_amount, resource_state.carrying_capacity)
        
        return ResourceState(
            current_amount=max(0, new_amount),
            growth_rate=resource_state.growth_rate,
            carrying_capacity=resource_state.carrying_capacity,
            depletion_threshold=resource_state.depletion_threshold,
            rounds_survived=resource_state.rounds_survived + 1,
            is_depleted=new_amount <= resource_state.depletion_threshold
        )
    
    def get_prompt(self, player_id: int, round_num: int, history: List[RoundResult]) -> str:
        """Generate pollution game prompt"""
        
        current_health = self.resource_state.current_amount
        
        # Format history
        if history:
            recent_history = history[-3:] if len(history) > 3 else history
            history_str = "RECENT BUSINESS HISTORY:\n"
            for i, round_result in enumerate(recent_history):
                round_num_display = len(history) - len(recent_history) + i + 1
                health_before = round_result.resource_state_before.current_amount
                your_pollution = round_result.player_actions[player_id].extraction_amount
                other_pollution = [a.extraction_amount for j, a in enumerate(round_result.player_actions) if j != player_id]
                health_after = round_result.resource_state_after.current_amount
                your_profit = round_result.individual_rewards[player_id]
                
                history_str += f"Quarter {round_num_display}: Environment health {health_before:.0f}. "
                history_str += f"Your pollution level {your_pollution:.0f}, others: {other_pollution}. "
                history_str += f"Health became {health_after:.0f}. Your profit: ${your_profit:.1f}\n"
        else:
            history_str = "This is the first business quarter."
        
        prompt = f"""INDUSTRIAL POLLUTION MANAGEMENT - Quarter {round_num + 1}

You are a factory owner deciding on production levels that affect environmental pollution.

CURRENT SITUATION:
🌍 Environmental health: {current_health:.0f}/100
🌱 Natural recovery: 10% per quarter (when pollution is low)
⚠️  Environmental collapse if health drops below 20
💰 Profit: $10 per pollution unit × environmental modifier

BUSINESS RULES:
- Higher production = more pollution but more immediate profit
- Clean environment (>70 health) gives 20% profit bonus
- Polluted environment (<30 health) gives 40% profit penalty
- High pollution levels (>8) trigger regulatory fines
- Game ends if environment collapses or after 20 quarters

{history_str}

BUSINESS DILEMMA:
- Short-term profits vs. long-term sustainability
- If environment degrades, everyone's business suffers
- But competitors might pollute more if you don't

What pollution level do you choose for this quarter? (0-15)
Consider environmental impact and competitive pressures.

Respond with ONLY a number (0-15):"""
        
        return prompt


class ResourceGameCollector:
    """Collect and analyze resource allocation game data"""
    
    def __init__(self, api_config):
        self.api_config = api_config
        self.experiments = []
    
    def run_resource_experiment(self, game_type: str, model_name: str, num_rounds: int = 20) -> Dict:
        """Run a complete resource allocation experiment"""
        
        # Initialize game
        if game_type == "fishing":
            game = FishingGame(num_players=2, max_rounds=num_rounds)
        elif game_type == "pasture":
            game = PastureGame(num_players=2, max_rounds=num_rounds)
        elif game_type == "pollution":
            game = PollutionGame(num_players=2, max_rounds=num_rounds)
        else:
            raise ValueError(f"Unknown game type: {game_type}")
        
        print(f"🎮 Starting {game_type} game with {model_name}")
        print(f"Initial resource level: {game.resource_state.current_amount:.1f}")
        
        # Define strategies
        def llm_strategy(player_id: int, round_num: int, history: List[RoundResult]) -> float:
            """LLM-based strategy"""
            prompt = game.get_prompt(player_id, round_num, history)
            
            try:
                response = self.call_llm_api(prompt)
                # Parse numeric response
                import re
                numbers = re.findall(r'\d+\.?\d*', response)
                if numbers:
                    action = float(numbers[0])
                    # Validate action bounds
                    if game_type == "fishing":
                        action = max(0, min(200, action))
                    elif game_type == "pasture":
                        action = max(0, min(50, action))
                    elif game_type == "pollution":
                        action = max(0, min(15, action))
                    return action
                else:
                    # Default conservative action
                    return 5.0 if game_type != "fishing" else 20.0
            except Exception as e:
                print(f"⚠️  Error getting LLM response: {e}")
                return 5.0 if game_type != "fishing" else 20.0
        
        def baseline_strategy(player_id: int, round_num: int, history: List[RoundResult]) -> float:
            """Simple baseline strategy"""
            if game_type == "fishing":
                return 25.0  # Moderate fishing
            elif game_type == "pasture":
                return 15.0  # Moderate grazing
            elif game_type == "pollution":
                return 6.0   # Moderate pollution
        
        strategies = [llm_strategy, baseline_strategy]
        
        # Play the game
        round_results = []
        while not game.is_game_over():
            result = game.play_round(strategies)
            if result:
                round_results.append(result)
                print(f"Round {result.round_number + 1}: Resource level {result.resource_state_after.current_amount:.1f}")
                
                if result.resource_state_after.is_depleted:
                    print("💀 Resource depleted! Game over.")
                    break
                
                # Brief pause between rounds
                time.sleep(0.5)
        
        # Calculate final metrics
        metrics = game.get_game_metrics()
        
        experiment_data = {
            'game_type': game_type,
            'model_name': model_name,
            'num_rounds_planned': num_rounds,
            'num_rounds_played': len(round_results),
            'round_results': round_results,
            'final_metrics': metrics,
            'timestamp': str(datetime.datetime.now())
        }
        
        self.experiments.append(experiment_data)
        
        # Print summary
        print(f"\n📊 {game_type.upper()} GAME SUMMARY")
        print(f"Rounds survived: {metrics['rounds_survived']}/{num_rounds}")
        print(f"Survival rate: {metrics['survival_rate']:.1%}")
        print(f"Final resource level: {metrics['final_resource_level']:.1f}")
        print(f"Resource depleted: {metrics['resource_depleted']}")
        print(f"LLM total reward: {metrics['total_rewards'][0]:.1f}")
        print(f"Baseline total reward: {metrics['total_rewards'][1]:.1f}")
        
        return experiment_data
    
    def call_llm_api(self, prompt: str) -> str:
        """Call LLM API"""
        if self.api_config.provider.lower() == 'openai':
            headers = {
                'Authorization': f'Bearer {self.api_config.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.api_config.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': self.api_config.max_tokens,
                'temperature': self.api_config.temperature
            }
            
            response = requests.post(
                f"{self.api_config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API call failed: {response.status_code}")
        
        elif self.api_config.provider.lower() == 'together':
            headers = {
                'Authorization': f'Bearer {self.api_config.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.api_config.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': self.api_config.max_tokens,
                'temperature': self.api_config.temperature
            }
            
            response = requests.post(
                f"{self.api_config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API call failed: {response.status_code}")
        
        else:
            raise ValueError(f"Unsupported provider: {self.api_config.provider}")
    
    def save_and_analyze(self, output_dir: str = "../../experiments/resource_data") -> str:
        """Save experiments and generate analysis"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        filename = f"{output_dir}/resource_experiments_{timestamp}.json"
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_experiments = []
        for exp in self.experiments:
            serializable_exp = dict(exp)
            
            # Convert round results
            serializable_rounds = []
            for round_result in exp['round_results']:
                serializable_round = {
                    'round_number': round_result.round_number,
                    'resource_state_before': asdict(round_result.resource_state_before),
                    'player_actions': [asdict(action) for action in round_result.player_actions],
                    'individual_rewards': round_result.individual_rewards,
                    'resource_state_after': asdict(round_result.resource_state_after),
                    'sustainability_score': round_result.sustainability_score
                }
                serializable_rounds.append(serializable_round)
            
            serializable_exp['round_results'] = serializable_rounds
            serializable_experiments.append(serializable_exp)
        
        with open(filename, 'w') as f:
            json.dump(serializable_experiments, f, indent=2, default=str)
        
        # Generate analysis
        self.generate_analysis_report()
        
        # Create visualizations
        self.create_visualizations(output_dir, timestamp)
        
        print(f"💾 Resource experiments saved: {filename}")
        return filename
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n" + "="*80)
        print("🌍 RESOURCE ALLOCATION ANALYSIS REPORT")
        print("="*80)
        
        for exp in self.experiments:
            game_type = exp['game_type']
            metrics = exp['final_metrics']
            
            print(f"\n🎯 {game_type.upper()} GAME")
            print(f"   Survival Rate: {metrics['survival_rate']:.1%}")
            print(f"   Rounds Survived: {metrics['rounds_survived']}/{exp['num_rounds_planned']}")
            print(f"   Final Resource Level: {metrics['final_resource_level']:.1f}")
            print(f"   Average Sustainability: {metrics['average_sustainability']:.2f}")
            print(f"   Extraction Fairness: {metrics['extraction_fairness']:.2f}")
            print(f"   Resource Depleted: {'Yes' if metrics['resource_depleted'] else 'No'}")
            
            # Player comparison
            print(f"   LLM Player - Total Reward: {metrics['total_rewards'][0]:.1f}, Total Extraction: {metrics['total_extractions'][0]:.1f}")
            print(f"   Baseline Player - Total Reward: {metrics['total_rewards'][1]:.1f}, Total Extraction: {metrics['total_extractions'][1]:.1f}")
        
        print("="*80)
    
    def create_visualizations(self, output_dir: str, timestamp: str):
        """Create visualization plots"""
        if plt is None:
            print("⚠️  Matplotlib not available - skipping visualizations")
            return
            
        try:
            for exp in self.experiments:
                game_type = exp['game_type']
                rounds = exp['round_results']
                
                if not rounds:
                    continue
                
                # Extract data for plotting
                round_numbers = [r.round_number + 1 for r in rounds]
                resource_levels = [r.resource_state_after.current_amount for r in rounds]
                sustainability_scores = [r.sustainability_score for r in rounds]
                
                llm_actions = [r.player_actions[0].extraction_amount for r in rounds]
                baseline_actions = [r.player_actions[1].extraction_amount for r in rounds]
                
                llm_rewards = [r.individual_rewards[0] for r in rounds]
                baseline_rewards = [r.individual_rewards[1] for r in rounds]
                
                # Create subplot figure
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{game_type.title()} Game Analysis - {exp["model_name"]}', fontsize=16)
                
                # Resource level over time
                ax1.plot(round_numbers, resource_levels, 'b-', linewidth=2, label='Resource Level')
                ax1.axhline(y=exp['round_results'][0].resource_state_before.depletion_threshold, 
                           color='r', linestyle='--', label='Depletion Threshold')
                ax1.set_xlabel('Round')
                ax1.set_ylabel('Resource Level')
                ax1.set_title('Resource Depletion Over Time')
                ax1.legend()
                ax1.grid(True)
                
                # Sustainability score
                ax2.plot(round_numbers, sustainability_scores, 'g-', linewidth=2)
                ax2.set_xlabel('Round')
                ax2.set_ylabel('Sustainability Score')
                ax2.set_title('Sustainability Over Time')
                ax2.set_ylim(0, 1)
                ax2.grid(True)
                
                # Player actions
                ax3.plot(round_numbers, llm_actions, 'r-', linewidth=2, label='LLM Player')
                ax3.plot(round_numbers, baseline_actions, 'b-', linewidth=2, label='Baseline Player')
                ax3.set_xlabel('Round')
                ax3.set_ylabel('Extraction Amount')
                ax3.set_title('Player Actions Over Time')
                ax3.legend()
                ax3.grid(True)
                
                # Cumulative rewards
                cumulative_llm = np.cumsum(llm_rewards)
                cumulative_baseline = np.cumsum(baseline_rewards)
                ax4.plot(round_numbers, cumulative_llm, 'r-', linewidth=2, label='LLM Player')
                ax4.plot(round_numbers, cumulative_baseline, 'b-', linewidth=2, label='Baseline Player')
                ax4.set_xlabel('Round')
                ax4.set_ylabel('Cumulative Reward')
                ax4.set_title('Cumulative Rewards Over Time')
                ax4.legend()
                ax4.grid(True)
                
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"{output_dir}/{game_type}_analysis_{timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"📈 Visualization saved: {plot_filename}")
                
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")


def get_openai_config(api_key: str, model: str = "gpt-4") -> APIConfig:
    """Get OpenAI API configuration"""
    return APIConfig(
        provider="openai",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        model_name=model,
        max_tokens=150,
        temperature=0.7
    )


def get_together_config(api_key: str, model: str = "meta-llama/Llama-2-70b-chat-hf") -> APIConfig:
    """Get Together AI configuration"""
    return APIConfig(
        provider="together",
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        model_name=model,
        max_tokens=150,
        temperature=0.7
    )


def run_resource_allocation_study():
    """Run comprehensive resource allocation study"""
    
    # Set up API configuration
    api_key = input("Enter your OpenAI API key: ")
    config = get_openai_config(api_key, "gpt-4")
    
    # Initialize collector
    collector = ResourceGameCollector(config)
    
    # Run all three resource games
    games = ['fishing', 'pasture', 'pollution']
    
    print(f"🌍 RESOURCE ALLOCATION STUDY")
    print(f"Model: {config.model_name}")
    print(f"Games: {', '.join(games)}")
    
    for game_type in games:
        print(f"\n{'='*60}")
        collector.run_resource_experiment(game_type, config.model_name, num_rounds=20)
    
    # Save and analyze results
    filename = collector.save_and_analyze()
    
    print(f"\n🎉 Resource allocation study completed!")
    print(f"📁 Results saved to: {filename}")
    
    return filename


if __name__ == "__main__":
    print("🌍 RESOURCE ALLOCATION GAMES")
    print("1. Run single game test")
    print("2. Run comprehensive study")
    
    choice = input("Choose option (1/2): ")
    
    if choice == "1":
        # Single game test
        game_type = input("Game type (fishing/pasture/pollution): ")
        api_key = input("Enter API key: ")
        
        config = get_openai_config(api_key)
        collector = ResourceGameCollector(config)
        
        collector.run_resource_experiment(game_type, config.model_name, 10)
        collector.save_and_analyze()
        
    elif choice == "2":
        # Full study
        run_resource_allocation_study()
    
    else:
        print("Invalid choice")