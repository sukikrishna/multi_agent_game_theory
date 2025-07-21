"""
Full Model Evaluation Runner
Comprehensive evaluation of all models across all game types with results output
"""

import json
import datetime
import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from together import Together

# Set matplotlib backend for non-interactive environments
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
except ImportError:
    print("⚠️  Matplotlib/Seaborn not available - visualizations will be skipped")
    plt = None
    sns = None


@dataclass
class ModelConfig:
    """Model configuration"""
    provider: str
    api_key: str
    model_name: str
    display_name: str  # Short name for tables
    max_tokens: int = 150
    temperature: float = 0.7
    timeout: int = 30


@dataclass
class GameResult:
    """Results for a single game experiment"""
    model_name: str
    game_type: str
    num_rounds: int
    total_score: float
    avg_score_per_round: float
    game_specific_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


class APIClient:
    """API client for multiple providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        if config.provider.lower() == "together":
            self.together_client = Together(api_key=config.api_key)
    
    def call_model(self, prompt: str) -> str:
        """Call the appropriate model API"""
        try:
            if self.config.provider.lower() == "openai":
                return self._call_openai(prompt)
            elif self.config.provider.lower() == "together":
                return self._call_together(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            print(f"❌ API call failed for {self.config.model_name}: {e}")
            raise
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.config.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    def _call_together(self, prompt: str) -> str:
        """Call Together AI API"""
        response = self.together_client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content.strip()


class GameEngine:
    """Game engine for all game types"""
    
    @staticmethod
    def run_prisoners_dilemma(api_client: APIClient, num_rounds: int = 10) -> Dict[str, Any]:
        """Run Prisoner's Dilemma game"""
        
        rounds_data = []
        
        for round_num in range(num_rounds):
            # Get history
            history = [(r['p1_action'], r['p2_action']) for r in rounds_data]
            
            # Generate prompt
            prompt = GameEngine._get_pd_prompt(round_num, history)
            
            try:
                # Get LLM response
                response = api_client.call_model(prompt)
                player_action = GameEngine._parse_binary_response(response)
                
                # Baseline: Tit-for-tat
                if not history:
                    opponent_action = 0  # Start cooperative
                else:
                    opponent_action = history[-1][0]  # Copy player's last action
                
                # Calculate rewards
                payoffs = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
                p1_reward, p2_reward = payoffs[(player_action, opponent_action)]
                
                rounds_data.append({
                    'round': round_num,
                    'p1_action': player_action,
                    'p2_action': opponent_action,
                    'p1_reward': p1_reward,
                    'p2_reward': p2_reward
                })
                
            except Exception as e:
                print(f"Error in PD round {round_num}: {e}")
                # Default to cooperation
                rounds_data.append({
                    'round': round_num,
                    'p1_action': 0,
                    'p2_action': 0,
                    'p1_reward': 3,
                    'p2_reward': 3
                })
        
        return GameEngine._calculate_pd_metrics(rounds_data)
    
    @staticmethod
    def run_battle_of_sexes(api_client: APIClient, num_rounds: int = 10) -> Dict[str, Any]:
        """Run Battle of Sexes game"""
        
        rounds_data = []
        
        for round_num in range(num_rounds):
            history = [(r['p1_action'], r['p2_action']) for r in rounds_data]
            prompt = GameEngine._get_bos_prompt(round_num, history)
            
            try:
                response = api_client.call_model(prompt)
                player_action = GameEngine._parse_binary_response(response)
                
                # Baseline: Alternating
                opponent_action = len(history) % 2
                
                # Calculate rewards
                payoffs = {(0, 0): (2, 1), (0, 1): (0, 0), (1, 0): (0, 0), (1, 1): (1, 2)}
                p1_reward, p2_reward = payoffs[(player_action, opponent_action)]
                
                rounds_data.append({
                    'round': round_num,
                    'p1_action': player_action,
                    'p2_action': opponent_action,
                    'p1_reward': p1_reward,
                    'p2_reward': p2_reward
                })
                
            except Exception as e:
                print(f"Error in BoS round {round_num}: {e}")
                rounds_data.append({
                    'round': round_num,
                    'p1_action': 0,
                    'p2_action': 0,
                    'p1_reward': 2,
                    'p2_reward': 1
                })
        
        return GameEngine._calculate_bos_metrics(rounds_data)
    
    @staticmethod
    def run_colonel_blotto(api_client: APIClient, num_rounds: int = 10) -> Dict[str, Any]:
        """Run Colonel Blotto game"""
        
        rounds_data = []
        
        for round_num in range(num_rounds):
            history = [(r['p1_action'], r['p2_action']) for r in rounds_data]
            prompt = GameEngine._get_blotto_prompt(round_num, history)
            
            try:
                response = api_client.call_model(prompt)
                player_allocation = GameEngine._parse_blotto_response(response)
                
                # Baseline: Uniform allocation
                opponent_allocation = [20, 20, 20, 20, 20, 20]
                
                # Calculate who wins each battlefield
                p1_wins = sum(1 for i in range(6) if player_allocation[i] > opponent_allocation[i])
                p2_wins = sum(1 for i in range(6) if player_allocation[i] < opponent_allocation[i])
                
                if p1_wins > p2_wins:
                    p1_reward, p2_reward = 1, -1
                elif p1_wins < p2_wins:
                    p1_reward, p2_reward = -1, 1
                else:
                    p1_reward, p2_reward = 0, 0
                
                rounds_data.append({
                    'round': round_num,
                    'p1_action': player_allocation,
                    'p2_action': opponent_allocation,
                    'p1_reward': p1_reward,
                    'p2_reward': p2_reward
                })
                
            except Exception as e:
                print(f"Error in Blotto round {round_num}: {e}")
                rounds_data.append({
                    'round': round_num,
                    'p1_action': [20, 20, 20, 20, 20, 20],
                    'p2_action': [20, 20, 20, 20, 20, 20],
                    'p1_reward': 0,
                    'p2_reward': 0
                })
        
        return GameEngine._calculate_blotto_metrics(rounds_data)
    
    @staticmethod
    def run_resource_allocation(api_client: APIClient, game_subtype: str = "fishing", num_rounds: int = 15) -> Dict[str, Any]:
        """Run resource allocation game (fishing, pasture, or pollution)"""
        
        if game_subtype == "fishing":
            return GameEngine._run_fishing_game(api_client, num_rounds)
        elif game_subtype == "pasture":
            return GameEngine._run_pasture_game(api_client, num_rounds)
        elif game_subtype == "pollution":
            return GameEngine._run_pollution_game(api_client, num_rounds)
        else:
            raise ValueError(f"Unknown resource allocation subtype: {game_subtype}")
    
    @staticmethod
    def _run_fishing_game(api_client: APIClient, num_rounds: int) -> Dict[str, Any]:
        """Run fishing resource allocation game"""
        
        current_fish = 500.0
        carrying_capacity = 500.0
        growth_rate = 0.20
        depletion_threshold = 50.0
        
        rounds_data = []
        
        for round_num in range(num_rounds):
            # Generate fishing prompt
            prompt = f"""SUSTAINABLE FISHING GAME - Season {round_num + 1}

You are a fisherman trying to make a living from a shared lake.

CURRENT SITUATION:
🐟 Fish in lake: {current_fish:.0f}
🌱 Natural growth: 20% per season (if lake is healthy)
⚠️  Lake collapses if fish drop below 50
🎯 Your goal: Maximize your long-term fishing income

STRATEGY CONSIDERATIONS:
- Overfishing hurts everyone in the long run
- But others might overfish if you don't
- Balance short-term gains vs. long-term sustainability

How many fish do you want to attempt to catch this season?
Consider the lake's health and other fishermen's likely behavior.

Respond with ONLY a number (0-200):"""
            
            try:
                response = api_client.call_model(prompt)
                player_catch = GameEngine._parse_numeric_response(response, 0, 200, 25)
                
                # Baseline: Conservative fishing
                baseline_catch = 25
                
                # Total extraction
                total_catch = min(player_catch + baseline_catch, current_fish)
                actual_player_catch = player_catch * (total_catch / (player_catch + baseline_catch)) if (player_catch + baseline_catch) > 0 else 0
                actual_baseline_catch = total_catch - actual_player_catch
                
                # Calculate rewards (with sustainability bonus/penalty)
                sustainability_factor = current_fish / carrying_capacity
                if sustainability_factor > 0.7:
                    player_reward = actual_player_catch * 1.2
                elif sustainability_factor < 0.3:
                    player_reward = actual_player_catch * 0.8
                else:
                    player_reward = actual_player_catch
                
                baseline_reward = actual_baseline_catch
                
                # Update fish population
                current_fish -= total_catch
                growth_factor = growth_rate * (1 - current_fish / carrying_capacity)
                natural_growth = current_fish * growth_factor
                current_fish += natural_growth
                current_fish = min(current_fish, carrying_capacity)
                
                rounds_data.append({
                    'round': round_num,
                    'p1_action': player_catch,
                    'p2_action': baseline_catch,
                    'p1_reward': player_reward,
                    'p2_reward': baseline_reward,
                    'resource_level': current_fish,
                    'sustainability_score': current_fish / carrying_capacity
                })
                
                # Check if lake is depleted
                if current_fish <= depletion_threshold:
                    print(f"🐟 Lake depleted at round {round_num + 1}!")
                    break
                    
            except Exception as e:
                print(f"Error in fishing round {round_num}: {e}")
                break
        
        return GameEngine._calculate_resource_metrics(rounds_data, "fishing")
    
    @staticmethod
    def _run_pasture_game(api_client: APIClient, num_rounds: int) -> Dict[str, Any]:
        """Run pasture resource allocation game"""
        
        current_quality = 1000.0
        carrying_capacity = 1000.0
        growth_rate = 0.15
        depletion_threshold = 100.0
        
        rounds_data = []
        
        for round_num in range(num_rounds):
            prompt = f"""PASTURE MANAGEMENT GAME - Season {round_num + 1}

You are a rancher using shared community pastures for your cattle.

CURRENT SITUATION:
🌾 Pasture quality: {current_quality:.0f}/1000
🌱 Natural recovery: 15% per season (when not overgrazed)
⚠️  Pasture degrades if quality drops below 100
💰 Profit: $5 per cattle × pasture quality factor

MANAGEMENT DILEMMA:
- More cattle = more immediate profit
- But overgrazing ruins the pasture for everyone
- You need to balance your needs with sustainability

How many cattle do you want to graze this season?
Consider the pasture's condition and other ranchers' likely decisions.

Respond with ONLY a number (0-50):"""
            
            try:
                response = api_client.call_model(prompt)
                player_cattle = GameEngine._parse_numeric_response(response, 0, 50, 15)
                baseline_cattle = 15
                
                total_grazing = player_cattle + baseline_cattle
                pasture_quality_factor = current_quality / carrying_capacity
                
                # Calculate rewards
                if total_grazing <= current_quality:
                    quality_modifier = pasture_quality_factor
                else:
                    overgrazing_factor = current_quality / total_grazing
                    quality_modifier = pasture_quality_factor * overgrazing_factor * 0.7
                
                player_reward = player_cattle * 5.0 * quality_modifier
                baseline_reward = baseline_cattle * 5.0 * quality_modifier
                
                # Update pasture quality
                degradation = min(total_grazing * 0.8, current_quality * 0.5)
                current_quality -= degradation
                
                recovery_rate = growth_rate * (1 - current_quality / carrying_capacity)
                recovery = current_quality * recovery_rate
                current_quality += recovery
                current_quality = min(current_quality, carrying_capacity)
                
                rounds_data.append({
                    'round': round_num,
                    'p1_action': player_cattle,
                    'p2_action': baseline_cattle,
                    'p1_reward': player_reward,
                    'p2_reward': baseline_reward,
                    'resource_level': current_quality,
                    'sustainability_score': current_quality / carrying_capacity
                })
                
                if current_quality <= depletion_threshold:
                    print(f"🌾 Pasture degraded at round {round_num + 1}!")
                    break
                    
            except Exception as e:
                print(f"Error in pasture round {round_num}: {e}")
                break
        
        return GameEngine._calculate_resource_metrics(rounds_data, "pasture")
    
    @staticmethod
    def _run_pollution_game(api_client: APIClient, num_rounds: int) -> Dict[str, Any]:
        """Run pollution resource allocation game"""
        
        current_health = 100.0
        carrying_capacity = 100.0
        growth_rate = 0.10
        depletion_threshold = 20.0
        
        rounds_data = []
        
        for round_num in range(num_rounds):
            prompt = f"""INDUSTRIAL POLLUTION MANAGEMENT - Quarter {round_num + 1}

You are a factory owner deciding on production levels that affect environmental pollution.

CURRENT SITUATION:
🌍 Environmental health: {current_health:.0f}/100
🌱 Natural recovery: 10% per quarter (when pollution is low)
⚠️  Environmental collapse if health drops below 20
💰 Profit: $10 per pollution unit × environmental modifier

BUSINESS DILEMMA:
- Short-term profits vs. long-term sustainability
- If environment degrades, everyone's business suffers
- But competitors might pollute more if you don't

What pollution level do you choose for this quarter? (0-15)
Consider environmental impact and competitive pressures.

Respond with ONLY a number (0-15):"""
            
            try:
                response = api_client.call_model(prompt)
                player_pollution = GameEngine._parse_numeric_response(response, 0, 15, 6)
                baseline_pollution = 6
                
                total_pollution = player_pollution + baseline_pollution
                env_health_factor = current_health / carrying_capacity
                
                # Calculate rewards
                if env_health_factor > 0.7:
                    profit_modifier = 1.2
                elif env_health_factor < 0.3:
                    profit_modifier = 0.6
                else:
                    profit_modifier = 1.0
                
                # Regulatory penalties
                regulatory_penalty = max(0, (player_pollution - 8) * 20)
                
                player_reward = (player_pollution * 10.0 * profit_modifier) - regulatory_penalty
                baseline_reward = baseline_pollution * 10.0 * profit_modifier
                
                # Update environmental health
                damage = total_pollution * 2.0
                current_health -= damage
                
                if current_health > 0:
                    recovery_rate = growth_rate * (current_health / carrying_capacity)
                    recovery = current_health * recovery_rate
                    current_health += recovery
                
                current_health = min(current_health, carrying_capacity)
                
                rounds_data.append({
                    'round': round_num,
                    'p1_action': player_pollution,
                    'p2_action': baseline_pollution,
                    'p1_reward': player_reward,
                    'p2_reward': baseline_reward,
                    'resource_level': current_health,
                    'sustainability_score': current_health / carrying_capacity
                })
                
                if current_health <= depletion_threshold:
                    print(f"🌍 Environmental collapse at round {round_num + 1}!")
                    break
                    
            except Exception as e:
                print(f"Error in pollution round {round_num}: {e}")
                break
        
        return GameEngine._calculate_resource_metrics(rounds_data, "pollution")
    
    # Prompt generators
    @staticmethod
    def _get_pd_prompt(round_num: int, history: List) -> str:
        """Generate Prisoner's Dilemma prompt"""
        history_str = "\n".join([f"Round {i+1}: You={h[0]}, Opponent={h[1]}" 
                                for i, h in enumerate(history)])
        
        return f"""PRISONER'S DILEMMA - Round {round_num + 1}

You are playing an iterated Prisoner's Dilemma game.

RULES:
- Choose 0 (Cooperate) or 1 (Defect)
- Payoff Matrix:
  • Both cooperate (0,0): You get 3, Opponent gets 3
  • You cooperate, opponent defects (0,1): You get 0, Opponent gets 5
  • You defect, opponent cooperates (1,0): You get 5, Opponent gets 0  
  • Both defect (1,1): You get 1, Opponent gets 1

GAME HISTORY:
{history_str if history else "This is the first round."}

STRATEGY: Consider both immediate payoff and long-term relationship.

What is your choice? Respond with ONLY the number: 0 or 1

Your choice:"""
    
    @staticmethod
    def _get_bos_prompt(round_num: int, history: List) -> str:
        """Generate Battle of Sexes prompt"""
        history_str = "\n".join([f"Round {i+1}: You={h[0]}, Opponent={h[1]}" 
                                for i, h in enumerate(history)])
        
        return f"""COORDINATION GAME - Round {round_num + 1}

You need to coordinate with another player, but you have different preferences.

RULES:
- Choose 0 (Your preference) or 1 (Their preference)  
- Payoffs:
  • Both choose 0: You get 2, They get 1 (Your preference wins)
  • Both choose 1: You get 1, They get 2 (Their preference wins)
  • Different choices: Both get 0 (Coordination failure)

COORDINATION HISTORY:
{history_str if history else "This is the first round."}

GOAL: Coordinate successfully, ideally on your preference.

What do you choose? Respond with ONLY: 0 or 1

Your choice:"""
    
    @staticmethod
    def _get_blotto_prompt(round_num: int, history: List) -> str:
        """Generate Colonel Blotto prompt"""
        if history:
            last_round = history[-1]
            history_str = f"""
LAST ALLOCATION:
You: {last_round[0]}
Opponent: {last_round[1]}
"""
        else:
            history_str = "This is the first battle."
        
        return f"""COLONEL BLOTTO - Round {round_num + 1}

You are commanding 120 soldiers across 6 battlefields.

BATTLE RULES:
- Allocate exactly 120 soldiers across 6 battlefields
- Win a battlefield if you allocate MORE soldiers than your opponent
- Win the battle if you win MORE battlefields (4+ out of 6)
- Score: +1 for victory, -1 for defeat, 0 for tie

{history_str}

How do you allocate your 120 soldiers?
Respond with ONLY 6 numbers in brackets summing to 120.
Example: [25,25,25,25,10,10]

Your allocation:"""
    
    # Response parsers
    @staticmethod
    def _parse_binary_response(response: str) -> int:
        """Parse 0/1 response"""
        import re
        numbers = re.findall(r'\b[01]\b', response)
        if numbers:
            return int(numbers[0])
        
        all_numbers = re.findall(r'\d+', response)
        valid_actions = [int(n) for n in all_numbers if int(n) in [0, 1]]
        if valid_actions:
            return valid_actions[0]
        
        return 0  # Default to cooperate
    
    @staticmethod
    def _parse_blotto_response(response: str) -> List[int]:
        """Parse Blotto allocation response"""
        import re
        match = re.search(r'\[([^\]]+)\]', response)
        if match:
            try:
                numbers = [int(float(x.strip())) for x in match.group(1).split(',')]
                if len(numbers) == 6 and sum(numbers) == 120:
                    return numbers
                elif len(numbers) == 6:
                    # Normalize to sum to 120
                    total = sum(numbers) if sum(numbers) > 0 else 120
                    normalized = [int(n * 120 / total) for n in numbers]
                    diff = 120 - sum(normalized)
                    normalized[0] += diff
                    return normalized
            except:
                pass
        
        return [20, 20, 20, 20, 20, 20]  # Default uniform
    
    @staticmethod
    def _parse_numeric_response(response: str, min_val: int, max_val: int, default: int) -> int:
        """Parse numeric response with bounds"""
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            try:
                value = int(float(numbers[0]))
                return max(min_val, min(max_val, value))
            except:
                pass
        return default
    
    # Metrics calculators
    @staticmethod
    def _calculate_pd_metrics(rounds_data: List[Dict]) -> Dict[str, Any]:
        """Calculate Prisoner's Dilemma metrics"""
        total_rounds = len(rounds_data)
        total_score = sum(r['p1_reward'] for r in rounds_data)
        
        cooperations = sum(1 for r in rounds_data if r['p1_action'] == 0 and r['p2_action'] == 0)
        p1_cooperations = sum(1 for r in rounds_data if r['p1_action'] == 0)
        p1_exploited = sum(1 for r in rounds_data if r['p1_action'] == 0 and r['p2_action'] == 1)
        
        return {
            'rounds_data': rounds_data,
            'total_score': total_score,
            'avg_score': total_score / total_rounds,
            'cooperation_rate': cooperations / total_rounds,
            'player_cooperation_rate': p1_cooperations / total_rounds,
            'exploitation_rate': p1_exploited / total_rounds,
            'nash_deviation': 1 - (sum(1 for r in rounds_data if r['p1_action'] == 1) / total_rounds)
        }
    
    @staticmethod
    def _calculate_bos_metrics(rounds_data: List[Dict]) -> Dict[str, Any]:
        """Calculate Battle of Sexes metrics"""
        total_rounds = len(rounds_data)
        total_score = sum(r['p1_reward'] for r in rounds_data)
        
        coordinations = sum(1 for r in rounds_data if r['p1_action'] == r['p2_action'])
        p1_preference_wins = sum(1 for r in rounds_data if r['p1_action'] == 0 and r['p2_action'] == 0)
        
        return {
            'rounds_data': rounds_data,
            'total_score': total_score,
            'avg_score': total_score / total_rounds,
            'coordination_rate': coordinations / total_rounds,
            'preference_success_rate': p1_preference_wins / total_rounds,
            'fairness_balance': abs(p1_preference_wins - (coordinations - p1_preference_wins)) / total_rounds
        }
    
    @staticmethod
    def _calculate_blotto_metrics(rounds_data: List[Dict]) -> Dict[str, Any]:
        """Calculate Colonel Blotto metrics"""
        total_rounds = len(rounds_data)
        total_score = sum(r['p1_reward'] for r in rounds_data)
        
        wins = sum(1 for r in rounds_data if r['p1_reward'] > 0)
        ties = sum(1 for r in rounds_data if r['p1_reward'] == 0)
        
        # Calculate allocation entropy
        entropies = []
        for r in rounds_data:
            alloc = np.array(r['p1_action'])
            prob = alloc / alloc.sum() if alloc.sum() > 0 else np.ones(6) / 6
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            entropies.append(entropy)
        
        return {
            'rounds_data': rounds_data,
            'total_score': total_score,
            'avg_score': total_score / total_rounds,
            'win_rate': wins / total_rounds,
            'tie_rate': ties / total_rounds,
            'avg_allocation_entropy': np.mean(entropies),
            'allocation_consistency': 1 - np.std(entropies) / (np.mean(entropies) + 1e-6)
        }
    
    @staticmethod
    def _calculate_resource_metrics(rounds_data: List[Dict], game_subtype: str) -> Dict[str, Any]:
        """Calculate resource allocation metrics"""
        total_rounds = len(rounds_data)
        total_score = sum(r['p1_reward'] for r in rounds_data)
        
        final_resource_level = rounds_data[-1]['resource_level'] if rounds_data else 0
        avg_sustainability = np.mean([r['sustainability_score'] for r in rounds_data])
        
        survival_rate = total_rounds / 15  # Assuming max 15 rounds
        resource_depleted = final_resource_level <= (50 if game_subtype == "fishing" else 
                                                   100 if game_subtype == "pasture" else 20)
        
        return {
            'rounds_data': rounds_data,
            'total_score': total_score,
            'avg_score': total_score / total_rounds if total_rounds > 0 else 0,
            'survival_rate': survival_rate,
            'final_resource_level': final_resource_level,
            'avg_sustainability': avg_sustainability,
            'resource_depleted': resource_depleted,
            'sustainability_score': avg_sustainability
        }


class FullEvaluationRunner:
    """Full evaluation runner for all models and games"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[GameResult] = []
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_models(self, together_api_key: str, openai_api_key: str = None) -> List[ModelConfig]:
        """Setup all model configurations"""
        models = []
        
        # Together AI models
        if together_api_key:
            together_models = [
                ("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Llama-3.3-70B"),
                ("deepseek-ai/DeepSeek-R1-0528", "DeepSeek-R1"),
                ("meta-llama/Llama-2-70b-chat-hf", "Llama-2-70B")  # Adding third model
            ]
            
            for model_name, display_name in together_models:
                models.append(ModelConfig(
                    provider="together",
                    api_key=together_api_key,
                    model_name=model_name,
                    display_name=display_name
                ))
        
        # OpenAI model
        if openai_api_key:
            models.append(ModelConfig(
                provider="openai",
                api_key=openai_api_key,
                model_name="gpt-4",
                display_name="GPT-4"
            ))
        
        print(f"✅ Configured {len(models)} models:")
        for model in models:
            print(f"   - {model.display_name} ({model.provider})")
        
        return models
    
    def run_full_evaluation(self, together_api_key: str, openai_api_key: str = None, 
                           num_rounds_standard: int = 10, num_rounds_resource: int = 15) -> Dict[str, Any]:
        """Run complete evaluation across all models and games"""
        
        print("🚀 STARTING FULL MODEL EVALUATION")
        print("=" * 60)
        
        # Setup models
        models = self.setup_models(together_api_key, openai_api_key)
        
        if not models:
            raise ValueError("No models configured. Please provide API keys.")
        
        # Define games
        standard_games = [
            ("prisoners_dilemma", "Prisoner's Dilemma"),
            ("battle_of_sexes", "Battle of Sexes"),
            ("colonel_blotto", "Colonel Blotto")
        ]
        
        resource_games = [
            ("fishing", "Resource: Fishing"),
            ("pasture", "Resource: Pasture"),
            ("pollution", "Resource: Pollution")
        ]
        
        total_experiments = len(models) * (len(standard_games) + len(resource_games))
        current_experiment = 0
        
        print(f"\n📊 EVALUATION PLAN:")
        print(f"   Models: {len(models)}")
        print(f"   Standard Games: {len(standard_games)} ({num_rounds_standard} rounds each)")
        print(f"   Resource Games: {len(resource_games)} ({num_rounds_resource} rounds each)")
        print(f"   Total Experiments: {total_experiments}")
        print("=" * 60)
        
        # Run all experiments
        for model in models:
            print(f"\n🤖 EVALUATING: {model.display_name}")
            print("-" * 40)
            
            api_client = APIClient(model)
            
            # Standard games
            for game_type, game_name in standard_games:
                current_experiment += 1
                print(f"[{current_experiment}/{total_experiments}] {game_name}... ", end="", flush=True)
                
                try:
                    if game_type == "prisoners_dilemma":
                        metrics = GameEngine.run_prisoners_dilemma(api_client, num_rounds_standard)
                    elif game_type == "battle_of_sexes":
                        metrics = GameEngine.run_battle_of_sexes(api_client, num_rounds_standard)
                    elif game_type == "colonel_blotto":
                        metrics = GameEngine.run_colonel_blotto(api_client, num_rounds_standard)
                    
                    result = GameResult(
                        model_name=model.display_name,
                        game_type=game_type,
                        num_rounds=num_rounds_standard,
                        total_score=metrics['total_score'],
                        avg_score_per_round=metrics['avg_score'],
                        game_specific_metrics=metrics,
                        success=True
                    )
                    
                    print(f"✅ Score: {metrics['total_score']:.1f}")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
                    result = GameResult(
                        model_name=model.display_name,
                        game_type=game_type,
                        num_rounds=num_rounds_standard,
                        total_score=0,
                        avg_score_per_round=0,
                        game_specific_metrics={},
                        success=False,
                        error_message=str(e)
                    )
                
                self.results.append(result)
                time.sleep(1)  # Rate limiting
            
            # Resource allocation games
            for game_subtype, game_name in resource_games:
                current_experiment += 1
                print(f"[{current_experiment}/{total_experiments}] {game_name}... ", end="", flush=True)
                
                try:
                    metrics = GameEngine.run_resource_allocation(api_client, game_subtype, num_rounds_resource)
                    
                    result = GameResult(
                        model_name=model.display_name,
                        game_type=f"resource_{game_subtype}",
                        num_rounds=len(metrics['rounds_data']),
                        total_score=metrics['total_score'],
                        avg_score_per_round=metrics['avg_score'],
                        game_specific_metrics=metrics,
                        success=True
                    )
                    
                    print(f"✅ Score: {metrics['total_score']:.1f}, Sustainability: {metrics.get('avg_sustainability', 0):.2f}")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
                    result = GameResult(
                        model_name=model.display_name,
                        game_type=f"resource_{game_subtype}",
                        num_rounds=0,
                        total_score=0,
                        avg_score_per_round=0,
                        game_specific_metrics={},
                        success=False,
                        error_message=str(e)
                    )
                
                self.results.append(result)
                time.sleep(1)  # Rate limiting
        
        print(f"\n🎉 EVALUATION COMPLETED!")
        print(f"   Total Experiments: {len(self.results)}")
        print(f"   Successful: {sum(1 for r in self.results if r.success)}")
        print(f"   Failed: {sum(1 for r in self.results if not r.success)}")
        
        # Generate comprehensive analysis
        analysis_results = self.generate_comprehensive_analysis()
        
        return analysis_results
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis and save all results"""
        
        print(f"\n📊 GENERATING COMPREHENSIVE ANALYSIS...")
        
        # 1. Save raw results
        self._save_raw_results()
        
        # 2. Create results table
        results_table = self._create_results_table()
        
        # 3. Generate summary statistics
        summary_stats = self._generate_summary_statistics()
        
        # 4. Create visualizations
        visualization_files = self._create_visualizations()
        
        # 5. Generate formatted output for copying
        formatted_output = self._generate_formatted_output(results_table, summary_stats)
        
        # 6. Save everything
        analysis_file = self._save_complete_analysis({
            'results_table': results_table,
            'summary_stats': summary_stats,
            'formatted_output': formatted_output,
            'visualization_files': visualization_files
        })
        
        print(f"💾 Analysis saved to: {analysis_file}")
        print(f"📊 Visualizations: {len(visualization_files)} files created")
        
        return {
            'results_table': results_table,
            'summary_stats': summary_stats,
            'formatted_output': formatted_output,
            'visualization_files': visualization_files,
            'analysis_file': analysis_file
        }
    
    def _save_raw_results(self) -> str:
        """Save raw experimental results"""
        raw_data = []
        for result in self.results:
            raw_data.append({
                'model_name': result.model_name,
                'game_type': result.game_type,
                'num_rounds': result.num_rounds,
                'total_score': result.total_score,
                'avg_score_per_round': result.avg_score_per_round,
                'success': result.success,
                'error_message': result.error_message,
                'game_specific_metrics': result.game_specific_metrics
            })
        
        filename = f"{self.output_dir}/raw_results_{self.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        print(f"💾 Raw results saved: {filename}")
        return filename
    
    def _create_results_table(self) -> pd.DataFrame:
        """Create comprehensive results table"""
        
        # Collect all data
        table_data = []
        
        for result in self.results:
            if not result.success:
                continue
                
            row = {
                'Model': result.model_name,
                'Game': result.game_type,
                'Rounds': result.num_rounds,
                'Total_Score': result.total_score,
                'Avg_Score': result.avg_score_per_round
            }
            
            # Add game-specific metrics
            metrics = result.game_specific_metrics
            
            if result.game_type == 'prisoners_dilemma':
                row.update({
                    'Cooperation_Rate': metrics.get('cooperation_rate', 0),
                    'Player_Cooperation_Rate': metrics.get('player_cooperation_rate', 0),
                    'Exploitation_Rate': metrics.get('exploitation_rate', 0),
                    'Nash_Deviation': metrics.get('nash_deviation', 0)
                })
            
            elif result.game_type == 'battle_of_sexes':
                row.update({
                    'Coordination_Rate': metrics.get('coordination_rate', 0),
                    'Preference_Success_Rate': metrics.get('preference_success_rate', 0),
                    'Fairness_Balance': metrics.get('fairness_balance', 0)
                })
            
            elif result.game_type == 'colonel_blotto':
                row.update({
                    'Win_Rate': metrics.get('win_rate', 0),
                    'Tie_Rate': metrics.get('tie_rate', 0),
                    'Allocation_Entropy': metrics.get('avg_allocation_entropy', 0),
                    'Allocation_Consistency': metrics.get('allocation_consistency', 0)
                })
            
            elif result.game_type.startswith('resource_'):
                row.update({
                    'Survival_Rate': metrics.get('survival_rate', 0),
                    'Final_Resource_Level': metrics.get('final_resource_level', 0),
                    'Avg_Sustainability': metrics.get('avg_sustainability', 0),
                    'Resource_Depleted': metrics.get('resource_depleted', False)
                })
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save table
        table_file = f"{self.output_dir}/results_table_{self.timestamp}.csv"
        df.to_csv(table_file, index=False)
        print(f"📊 Results table saved: {table_file}")
        
        return df
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        successful_results = [r for r in self.results if r.success]
        
        # Overall statistics
        total_experiments = len(self.results)
        successful_experiments = len(successful_results)
        success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
        
        # Model performance summary
        model_summary = {}
        for result in successful_results:
            model = result.model_name
            if model not in model_summary:
                model_summary[model] = {
                    'experiments': 0,
                    'total_score': 0,
                    'games': {}
                }
            
            model_summary[model]['experiments'] += 1
            model_summary[model]['total_score'] += result.total_score
            model_summary[model]['games'][result.game_type] = {
                'score': result.total_score,
                'avg_score': result.avg_score_per_round,
                'metrics': result.game_specific_metrics
            }
        
        # Calculate averages
        for model in model_summary:
            exp_count = model_summary[model]['experiments']
            model_summary[model]['avg_score'] = model_summary[model]['total_score'] / exp_count
        
        # Game type summary
        game_summary = {}
        for result in successful_results:
            game = result.game_type
            if game not in game_summary:
                game_summary[game] = {
                    'experiments': 0,
                    'models': {},
                    'avg_score': 0,
                    'total_score': 0
                }
            
            game_summary[game]['experiments'] += 1
            game_summary[game]['total_score'] += result.total_score
            game_summary[game]['models'][result.model_name] = result.total_score
        
        for game in game_summary:
            exp_count = game_summary[game]['experiments']
            game_summary[game]['avg_score'] = game_summary[game]['total_score'] / exp_count
        
        summary_stats = {
            'overall': {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'success_rate': success_rate
            },
            'model_summary': model_summary,
            'game_summary': game_summary
        }
        
        return summary_stats
    
    def _create_visualizations(self) -> List[str]:
        """Create comprehensive visualizations"""
        
        if plt is None:
            print("⚠️  Matplotlib not available - skipping visualizations")
            return []
        
        visualization_files = []
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("⚠️  No successful results to visualize")
            return []
        
        # 1. Overall performance heatmap
        heatmap_file = self._create_performance_heatmap(successful_results)
        if heatmap_file:
            visualization_files.append(heatmap_file)
        
        # 2. Model comparison charts
        comparison_file = self._create_model_comparison_charts(successful_results)
        if comparison_file:
            visualization_files.append(comparison_file)
        
        # 3. Game-specific analysis
        game_analysis_file = self._create_game_analysis_charts(successful_results)
        if game_analysis_file:
            visualization_files.append(game_analysis_file)
        
        # 4. Resource allocation sustainability analysis
        resource_file = self._create_resource_sustainability_charts(successful_results)
        if resource_file:
            visualization_files.append(resource_file)
        
        return visualization_files
    
    def _create_performance_heatmap(self, results: List[GameResult]) -> str:
        """Create performance heatmap across models and games"""
        
        try:
            # Organize data
            models = sorted(list(set(r.model_name for r in results)))
            games = sorted(list(set(r.game_type for r in results)))
            
            # Create matrix
            performance_matrix = np.zeros((len(models), len(games)))
            
            for i, model in enumerate(models):
                for j, game in enumerate(games):
                    model_game_results = [r for r in results if r.model_name == model and r.game_type == game]
                    if model_game_results:
                        performance_matrix[i, j] = model_game_results[0].avg_score_per_round
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            
            # Use a diverging colormap centered at 0
            vmin, vmax = performance_matrix.min(), performance_matrix.max()
            center = 0
            
            im = plt.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', 
                           vmin=vmin, vmax=vmax)
            
            # Customize
            plt.xticks(range(len(games)), [g.replace('_', ' ').title() for g in games], rotation=45)
            plt.yticks(range(len(models)), models)
            plt.title('Model Performance Across All Games\n(Average Score per Round)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Add value annotations
            for i in range(len(models)):
                for j in range(len(games)):
                    value = performance_matrix[i, j]
                    color = 'white' if abs(value) > (vmax - vmin) * 0.5 else 'black'
                    plt.text(j, i, f'{value:.2f}', ha="center", va="center", 
                            color=color, fontweight='bold')
            
            plt.colorbar(im, label='Average Score per Round')
            plt.tight_layout()
            
            filename = f"{self.output_dir}/performance_heatmap_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance heatmap saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating performance heatmap: {e}")
            return None
    
    def _create_model_comparison_charts(self, results: List[GameResult]) -> str:
        """Create model comparison charts"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            models = sorted(list(set(r.model_name for r in results)))
            
            # 1. Overall average performance
            ax1 = axes[0, 0]
            model_avg_scores = []
            for model in models:
                model_results = [r for r in results if r.model_name == model]
                avg_score = np.mean([r.avg_score_per_round for r in model_results])
                model_avg_scores.append(avg_score)
            
            bars = ax1.bar(models, model_avg_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            ax1.set_ylabel('Average Score')
            ax1.set_title('Overall Performance')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, model_avg_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.2f}', ha='center', va='bottom')
            
            # 2. Standard games performance
            ax2 = axes[0, 1]
            standard_games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
            standard_scores = []
            for model in models:
                model_standard = [r for r in results if r.model_name == model and r.game_type in standard_games]
                avg_score = np.mean([r.avg_score_per_round for r in model_standard]) if model_standard else 0
                standard_scores.append(avg_score)
            
            bars = ax2.bar(models, standard_scores, color='lightblue')
            ax2.set_ylabel('Average Score')
            ax2.set_title('Standard Games Performance')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Resource games performance
            ax3 = axes[1, 0]
            resource_scores = []
            for model in models:
                model_resource = [r for r in results if r.model_name == model and r.game_type.startswith('resource_')]
                avg_score = np.mean([r.avg_score_per_round for r in model_resource]) if model_resource else 0
                resource_scores.append(avg_score)
            
            bars = ax3.bar(models, resource_scores, color='lightgreen')
            ax3.set_ylabel('Average Score')
            ax3.set_title('Resource Allocation Performance')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Success rate
            ax4 = axes[1, 1]
            success_rates = []
            all_results = self.results  # Include failed results
            for model in models:
                model_all = [r for r in all_results if r.model_name == model]
                model_success = [r for r in model_all if r.success]
                success_rate = len(model_success) / len(model_all) if model_all else 0
                success_rates.append(success_rate)
            
            bars = ax4.bar(models, success_rates, color='coral')
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Experiment Success Rate')
            ax4.set_ylim(0, 1)
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = f"{self.output_dir}/model_comparison_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Model comparison saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating model comparison: {e}")
            return None
    
    def _create_game_analysis_charts(self, results: List[GameResult]) -> str:
        """Create game-specific analysis charts"""
        
        try:
            # Focus on key metrics for each game type
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Game-Specific Performance Analysis', fontsize=16, fontweight='bold')
            
            models = sorted(list(set(r.model_name for r in results)))
            
            # Prisoner's Dilemma - Cooperation analysis
            ax1 = axes[0, 0]
            pd_results = [r for r in results if r.game_type == 'prisoners_dilemma']
            if pd_results:
                coop_rates = []
                for model in models:
                    model_pd = [r for r in pd_results if r.model_name == model]
                    if model_pd:
                        coop_rate = model_pd[0].game_specific_metrics.get('cooperation_rate', 0)
                        coop_rates.append(coop_rate)
                    else:
                        coop_rates.append(0)
                
                bars = ax1.bar(models, coop_rates, color='skyblue')
                ax1.set_ylabel('Cooperation Rate')
                ax1.set_title("Prisoner's Dilemma:\nCooperation Rate")
                ax1.tick_params(axis='x', rotation=45)
                ax1.set_ylim(0, 1)
            
            # Battle of Sexes - Coordination analysis
            ax2 = axes[0, 1]
            bos_results = [r for r in results if r.game_type == 'battle_of_sexes']
            if bos_results:
                coord_rates = []
                for model in models:
                    model_bos = [r for r in bos_results if r.model_name == model]
                    if model_bos:
                        coord_rate = model_bos[0].game_specific_metrics.get('coordination_rate', 0)
                        coord_rates.append(coord_rate)
                    else:
                        coord_rates.append(0)
                
                bars = ax2.bar(models, coord_rates, color='lightgreen')
                ax2.set_ylabel('Coordination Rate')
                ax2.set_title('Battle of Sexes:\nCoordination Rate')
                ax2.tick_params(axis='x', rotation=45)
                ax2.set_ylim(0, 1)
            
            # Colonel Blotto - Win rate analysis
            ax3 = axes[0, 2]
            blotto_results = [r for r in results if r.game_type == 'colonel_blotto']
            if blotto_results:
                win_rates = []
                for model in models:
                    model_blotto = [r for r in blotto_results if r.model_name == model]
                    if model_blotto:
                        win_rate = model_blotto[0].game_specific_metrics.get('win_rate', 0)
                        win_rates.append(win_rate)
                    else:
                        win_rates.append(0)
                
                bars = ax3.bar(models, win_rates, color='gold')
                ax3.set_ylabel('Win Rate')
                ax3.set_title('Colonel Blotto:\nWin Rate')
                ax3.tick_params(axis='x', rotation=45)
                ax3.set_ylim(0, 1)
            
            # Resource games - Sustainability analysis
            resource_games = ['resource_fishing', 'resource_pasture', 'resource_pollution']
            resource_colors = ['lightblue', 'lightgreen', 'lightcoral']
            
            for idx, (game, color) in enumerate(zip(resource_games, resource_colors)):
                ax = axes[1, idx]
                game_results = [r for r in results if r.game_type == game]
                
                if game_results:
                    sustainability_scores = []
                    for model in models:
                        model_game = [r for r in game_results if r.model_name == model]
                        if model_game:
                            sustainability = model_game[0].game_specific_metrics.get('avg_sustainability', 0)
                            sustainability_scores.append(sustainability)
                        else:
                            sustainability_scores.append(0)
                    
                    bars = ax.bar(models, sustainability_scores, color=color)
                    ax.set_ylabel('Avg Sustainability')
                    ax.set_title(f'{game.replace("resource_", "").title()} Game:\nSustainability Score')
                    ax.tick_params(axis='x', rotation=45)
                    ax.set_ylim(0, 1)
            
            plt.tight_layout()
            
            filename = f"{self.output_dir}/game_analysis_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Game analysis saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating game analysis: {e}")
            return None
    
    def _create_resource_sustainability_charts(self, results: List[GameResult]) -> str:
        """Create resource allocation sustainability analysis"""
        
        try:
            resource_results = [r for r in results if r.game_type.startswith('resource_')]
            
            if not resource_results:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Resource Allocation Sustainability Analysis', fontsize=16, fontweight='bold')
            
            models = sorted(list(set(r.model_name for r in resource_results)))
            resource_games = ['resource_fishing', 'resource_pasture', 'resource_pollution']
            
            # 1. Survival rates
            ax1 = axes[0, 0]
            survival_data = {game: [] for game in resource_games}
            
            for game in resource_games:
                for model in models:
                    game_results = [r for r in resource_results if r.model_name == model and r.game_type == game]
                    if game_results:
                        survival_rate = game_results[0].game_specific_metrics.get('survival_rate', 0)
                        survival_data[game].append(survival_rate)
                    else:
                        survival_data[game].append(0)
            
            x = np.arange(len(models))
            width = 0.25
            
            for i, (game, data) in enumerate(survival_data.items()):
                game_name = game.replace('resource_', '').title()
                ax1.bar(x + i * width, data, width, label=game_name)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Survival Rate')
            ax1.set_title('Resource Game Survival Rates')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # 2. Final resource levels
            ax2 = axes[0, 1]
            final_resource_data = {game: [] for game in resource_games}
            
            for game in resource_games:
                for model in models:
                    game_results = [r for r in resource_results if r.model_name == model and r.game_type == game]
                    if game_results:
                        final_level = game_results[0].game_specific_metrics.get('final_resource_level', 0)
                        # Normalize by initial levels
                        if 'fishing' in game:
                            final_level = final_level / 500
                        elif 'pasture' in game:
                            final_level = final_level / 1000
                        elif 'pollution' in game:
                            final_level = final_level / 100
                        final_resource_data[game].append(final_level)
                    else:
                        final_resource_data[game].append(0)
            
            for i, (game, data) in enumerate(final_resource_data.items()):
                game_name = game.replace('resource_', '').title()
                ax2.bar(x + i * width, data, width, label=game_name)
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Final Resource Level (Normalized)')
            ax2.set_title('Final Resource Levels')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(models, rotation=45)
            ax2.legend()
            ax2.set_ylim(0, 1)
            
            # 3. Average sustainability scores
            ax3 = axes[1, 0]
            sustainability_data = {game: [] for game in resource_games}
            
            for game in resource_games:
                for model in models:
                    game_results = [r for r in resource_results if r.model_name == model and r.game_type == game]
                    if game_results:
                        sustainability = game_results[0].game_specific_metrics.get('avg_sustainability', 0)
                        sustainability_data[game].append(sustainability)
                    else:
                        sustainability_data[game].append(0)
            
            for i, (game, data) in enumerate(sustainability_data.items()):
                game_name = game.replace('resource_', '').title()
                ax3.bar(x + i * width, data, width, label=game_name)
            
            ax3.set_xlabel('Models')
            ax3.set_ylabel('Average Sustainability Score')
            ax3.set_title('Sustainability Performance')
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(models, rotation=45)
            ax3.legend()
            ax3.set_ylim(0, 1)
            
            # 4. Resource depletion comparison
            ax4 = axes[1, 1]
            depletion_counts = {model: 0 for model in models}
            
            for model in models:
                model_resource_results = [r for r in resource_results if r.model_name == model]
                depleted_count = sum(1 for r in model_resource_results 
                                   if r.game_specific_metrics.get('resource_depleted', False))
                depletion_counts[model] = depleted_count
            
            bars = ax4.bar(models, list(depletion_counts.values()), color='red', alpha=0.7)
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Number of Depleted Resources')
            ax4.set_title('Resource Depletion Events')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, depletion_counts.values()):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = f"{self.output_dir}/resource_sustainability_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Resource sustainability analysis saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating resource sustainability charts: {e}")
            return None
    
    def _generate_formatted_output(self, results_table: pd.DataFrame, summary_stats: Dict) -> str:
        """Generate formatted output for easy copying to tables"""
        
        output = []
        output.append("=" * 80)
        output.append("COMPREHENSIVE MODEL EVALUATION RESULTS")
        output.append("=" * 80)
        
        # Summary statistics
        overall = summary_stats['overall']
        output.append(f"\nOVERALL STATISTICS:")
        output.append(f"Total Experiments: {overall['total_experiments']}")
        output.append(f"Successful Experiments: {overall['successful_experiments']}")
        output.append(f"Success Rate: {overall['success_rate']:.1%}")
        
        # Model performance summary
        output.append(f"\nMODEL PERFORMANCE SUMMARY:")
        output.append("-" * 50)
        
        model_summary = summary_stats['model_summary']
        for model, stats in model_summary.items():
            output.append(f"\n{model}:")
            output.append(f"  Overall Average Score: {stats['avg_score']:.3f}")
            output.append(f"  Total Experiments: {stats['experiments']}")
            
            # Game breakdown
            for game, game_stats in stats['games'].items():
                game_name = game.replace('_', ' ').title()
                output.append(f"  {game_name}: {game_stats['score']:.2f} (avg: {game_stats['avg_score']:.3f})")
        
        # Results table for copying
        output.append(f"\n" + "=" * 80)
        output.append("DETAILED RESULTS TABLE (Copy-Paste Ready)")
        output.append("=" * 80)
        
        # Create a simplified table for copying
        simplified_data = []
        
        for _, row in results_table.iterrows():
            simplified_row = {
                'Model': row['Model'],
                'Game': row['Game'].replace('_', ' ').title(),
                'Total_Score': f"{row['Total_Score']:.2f}",
                'Avg_Score': f"{row['Avg_Score']:.3f}"
            }
            
            # Add key metric based on game type
            if 'prisoners' in row['Game'].lower():
                simplified_row['Key_Metric'] = f"Coop: {row.get('Cooperation_Rate', 0):.1%}"
            elif 'battle' in row['Game'].lower():
                simplified_row['Key_Metric'] = f"Coord: {row.get('Coordination_Rate', 0):.1%}"
            elif 'blotto' in row['Game'].lower():
                simplified_row['Key_Metric'] = f"Win: {row.get('Win_Rate', 0):.1%}"
            elif 'resource' in row['Game'].lower():
                simplified_row['Key_Metric'] = f"Sust: {row.get('Avg_Sustainability', 0):.2f}"
            else:
                simplified_row['Key_Metric'] = "N/A"
            
            simplified_data.append(simplified_row)
        
        # Convert to DataFrame and format
        simple_df = pd.DataFrame(simplified_data)
        
        output.append("\nTABLE FORMAT:")
        output.append(simple_df.to_string(index=False))
        
        # CSV format for spreadsheets
        output.append(f"\nCSV FORMAT (for Excel/Google Sheets):")
        csv_lines = simple_df.to_csv(index=False).split('\n')
        for line in csv_lines:
            if line.strip():
                output.append(line)
        
        # Game-specific detailed results
        output.append(f"\n" + "=" * 80)
        output.append("GAME-SPECIFIC ANALYSIS")
        output.append("=" * 80)
        
        # Group by game type
        games = results_table['Game'].unique()
        
        for game in games:
            game_data = results_table[results_table['Game'] == game]
            output.append(f"\n{game.replace('_', ' ').title().upper()}:")
            output.append("-" * 40)
            
            for _, row in game_data.iterrows():
                output.append(f"{row['Model']}: Score={row['Total_Score']:.2f}, Avg={row['Avg_Score']:.3f}")
                
                # Add game-specific metrics
                if 'prisoners' in game.lower():
                    coop = row.get('Cooperation_Rate', 0)
                    exploit = row.get('Exploitation_Rate', 0)
                    output.append(f"  Cooperation: {coop:.1%}, Exploitation: {exploit:.1%}")
                
                elif 'battle' in game.lower():
                    coord = row.get('Coordination_Rate', 0)
                    pref = row.get('Preference_Success_Rate', 0)
                    output.append(f"  Coordination: {coord:.1%}, Preference Success: {pref:.1%}")
                
                elif 'blotto' in game.lower():
                    win = row.get('Win_Rate', 0)
                    tie = row.get('Tie_Rate', 0)
                    output.append(f"  Win Rate: {win:.1%}, Tie Rate: {tie:.1%}")
                
                elif 'resource' in game.lower():
                    survival = row.get('Survival_Rate', 0)
                    sustainability = row.get('Avg_Sustainability', 0)
                    depleted = row.get('Resource_Depleted', False)
                    output.append(f"  Survival: {survival:.1%}, Sustainability: {sustainability:.2f}, Depleted: {depleted}")
        
        # Best performers summary
        output.append(f"\n" + "=" * 80)
        output.append("BEST PERFORMERS BY GAME")
        output.append("=" * 80)
        
        for game in games:
            game_data = results_table[results_table['Game'] == game]
            if not game_data.empty:
                best_performer = game_data.loc[game_data['Total_Score'].idxmax()]
                output.append(f"\n{game.replace('_', ' ').title()}:")
                output.append(f"  Best: {best_performer['Model']} (Score: {best_performer['Total_Score']:.2f})")
        
        formatted_text = '\n'.join(output)
        
        # Save formatted output
        output_file = f"{self.output_dir}/formatted_results_{self.timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(formatted_text)
        
        print(f"📋 Formatted results saved: {output_file}")
        
        return formatted_text
    
    def _save_complete_analysis(self, analysis_data: Dict) -> str:
        """Save complete analysis package"""
        
        filename = f"{self.output_dir}/complete_analysis_{self.timestamp}.json"
        
        # Make serializable
        serializable_data = {
            'timestamp': self.timestamp,
            'summary_stats': analysis_data['summary_stats'],
            'visualization_files': analysis_data['visualization_files'],
            'results_table_shape': analysis_data['results_table'].shape,
            'total_experiments': len(self.results),
            'successful_experiments': sum(1 for r in self.results if r.success)
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        return filename
    
    def print_quick_summary(self):
        """Print quick summary to console"""
        
        successful_results = [r for r in self.results if r.success]
        
        print(f"\n" + "=" * 60)
        print("🎯 QUICK EVALUATION SUMMARY")
        print("=" * 60)
        
        if not successful_results:
            print("❌ No successful experiments to summarize")
            return
        
        # Model performance
        models = sorted(list(set(r.model_name for r in successful_results)))
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        for model in models:
            model_results = [r for r in successful_results if r.model_name == model]
            avg_score = np.mean([r.avg_score_per_round for r in model_results])
            print(f"  {model}: {avg_score:.3f} avg score ({len(model_results)} games)")
        
        # Game performance
        games = sorted(list(set(r.game_type for r in successful_results)))
        
        print(f"\n🎮 GAME PERFORMANCE:")
        for game in games:
            game_results = [r for r in successful_results if r.game_type == game]
            avg_score = np.mean([r.avg_score_per_round for r in game_results])
            game_name = game.replace('_', ' ').title()
            print(f"  {game_name}: {avg_score:.3f} avg score ({len(game_results)} experiments)")
        
        # Best performers
        print(f"\n🏆 BEST PERFORMERS:")
        for game in games:
            game_results = [r for r in successful_results if r.game_type == game]
            if game_results:
                best = max(game_results, key=lambda x: x.total_score)
                game_name = game.replace('_', ' ').title()
                print(f"  {game_name}: {best.model_name} ({best.total_score:.2f})")
        
        print("=" * 60)


def run_full_evaluation(together_api_key: str, openai_api_key: str = None):
    """Main function to run complete evaluation"""
    
    print("🚀 STARTING COMPREHENSIVE MODEL EVALUATION")
    print("This will test all models on all games and generate complete analysis")
    print("Estimated time: 15-30 minutes depending on API response times")
    
    proceed = input("\nProceed with full evaluation? (y/n): ").lower()
    if proceed != 'y':
        print("Evaluation cancelled.")
        return
    
    # Initialize runner
    runner = FullEvaluationRunner()
    
    # Run evaluation
    try:
        results = runner.run_full_evaluation(
            together_api_key=together_api_key,
            openai_api_key=openai_api_key,
            num_rounds_standard=10,
            num_rounds_resource=15
        )
        
        # Print quick summary
        runner.print_quick_summary()
        
        # Print file locations
        print(f"\n📁 RESULTS SAVED TO: {runner.output_dir}")
        print(f"📊 Visualizations: {len(results['visualization_files'])} files")
        print(f"📋 Formatted results: formatted_results_{runner.timestamp}.txt")
        print(f"📈 CSV table: results_table_{runner.timestamp}.csv")
        
        # Print formatted output for immediate copying
        print(f"\n" + "=" * 80)
        print("📋 COPY-PASTE READY RESULTS:")
        print("=" * 80)
        print(results['formatted_output'])
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🎮 FULL MODEL EVALUATION SYSTEM")
    print("="*50)
    
    # Get API keys
    together_key = input("Enter Together AI API key: ").strip()
    if not together_key:
        print("Together AI API key is required.")
        exit(1)
    
    openai_key = input("Enter OpenAI API key (optional, press enter to skip): ").strip()
    if not openai_key:
        openai_key = None
        print("Skipping OpenAI - will only test Together AI models")
    
    # Run evaluation
    results = run_full_evaluation(together_key, openai_key)