"""
Enhanced Multi-Agent Game Theory System
Supports multiple API providers (OpenAI, Together AI, Anthropic Claude, etc.)
with comprehensive evaluation and visualization capabilities
"""

import json
import datetime
import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import requests
from together import Together

# Set matplotlib backend for non-interactive environments
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("⚠️  Matplotlib/Seaborn not available - visualizations will be skipped")
    plt = None
    sns = None


@dataclass
class ModelConfig:
    """Enhanced configuration for different API providers"""
    provider: str
    api_key: str
    model_name: str
    base_url: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.7
    timeout: int = 30
    
    def __post_init__(self):
        """Set default base URLs"""
        if self.base_url is None:
            if self.provider.lower() == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.provider.lower() == "together":
                self.base_url = "https://api.together.xyz/v1"
            elif self.provider.lower() == "anthropic":
                self.base_url = "https://api.anthropic.com/v1"


@dataclass
class ExperimentResult:
    """Results from a complete experiment"""
    experiment_id: str
    model_config: ModelConfig
    game_type: str
    num_rounds: int
    rounds_data: List[Dict]
    metrics: Dict[str, float]
    timestamp: str
    success: bool
    error_message: Optional[str] = None


class EnhancedAPIClient:
    """Enhanced API client supporting multiple providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._setup_client()
    
    def _setup_client(self):
        """Initialize provider-specific clients"""
        if self.config.provider.lower() == "together":
            # Initialize Together client
            self.together_client = Together(api_key=self.config.api_key)
        elif self.config.provider.lower() == "anthropic":
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                print("⚠️  Anthropic SDK not installed. Install with: pip install anthropic")
                self.anthropic_client = None
    
    def call_model(self, prompt: str) -> str:
        """Call the appropriate model API"""
        try:
            if self.config.provider.lower() == "openai":
                return self._call_openai(prompt)
            elif self.config.provider.lower() == "together":
                return self._call_together(prompt)
            elif self.config.provider.lower() == "anthropic":
                return self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            print(f"❌ API call failed: {e}")
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
            f"{self.config.base_url}/chat/completions",
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
        """Call Together AI API using the official SDK"""
        response = self.together_client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content.strip()
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        if self.anthropic_client is None:
            raise Exception("Anthropic client not initialized")
        
        response = self.anthropic_client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class GamePromptGenerator:
    """Enhanced prompt generator with multiple variations"""
    
    @staticmethod
    def get_prompt(game_type: str, round_num: int, history: List[Tuple], 
                   variant: str = "standard", model_name: str = "") -> str:
        """Generate game prompts with model-specific optimizations"""
        
        if game_type == "prisoners_dilemma":
            return GamePromptGenerator._get_pd_prompt(round_num, history, variant, model_name)
        elif game_type == "battle_of_sexes":
            return GamePromptGenerator._get_bos_prompt(round_num, history, variant, model_name)
        elif game_type == "colonel_blotto":
            return GamePromptGenerator._get_blotto_prompt(round_num, history, variant, model_name)
        else:
            raise ValueError(f"Unknown game type: {game_type}")
    
    @staticmethod
    def _get_pd_prompt(round_num: int, history: List[Tuple], variant: str, model_name: str) -> str:
        """Enhanced Prisoner's Dilemma prompt"""
        history_str = "\n".join([f"Round {i+1}: You={h[0]}, Opponent={h[1]}" 
                                for i, h in enumerate(history)])
        
        # Model-specific optimization
        instruction_style = "Think step by step." if "llama" in model_name.lower() else ""
        
        base_prompt = f"""PRISONER'S DILEMMA - Round {round_num + 1}

{instruction_style}

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
        
        return base_prompt
    
    @staticmethod
    def _get_bos_prompt(round_num: int, history: List[Tuple], variant: str, model_name: str) -> str:
        """Enhanced Battle of Sexes prompt"""
        history_str = "\n".join([f"Round {i+1}: You={h[0]}, Opponent={h[1]}" 
                                for i, h in enumerate(history)])
        
        base_prompt = f"""COORDINATION GAME - Round {round_num + 1}

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
        
        return base_prompt
    
    @staticmethod
    def _get_blotto_prompt(round_num: int, history: List[Tuple], variant: str, model_name: str) -> str:
        """Enhanced Colonel Blotto prompt"""
        if history:
            last_round = history[-1]
            history_str = f"""
LAST ALLOCATION:
You: {last_round[0]}
Opponent: {last_round[1]}

RECENT HISTORY (last 3 rounds):
""" + "\n".join([f"Round {len(history)-2+i}: You: {h[0]}, Opponent: {h[1]}" 
                 for i, h in enumerate(history[-3:])])
        else:
            history_str = "This is the first battle."
        
        base_prompt = f"""COLONEL BLOTTO - Round {round_num + 1}

You are commanding 120 soldiers across 6 battlefields.

BATTLE RULES:
- Allocate exactly 120 soldiers across 6 battlefields
- Win a battlefield if you allocate MORE soldiers than your opponent
- Win the battle if you win MORE battlefields (4+ out of 6)
- Score: +1 for victory, -1 for defeat, 0 for tie

ALLOCATION STRATEGIES:
- Uniform: [20,20,20,20,20,20] - Balanced approach
- Concentrated: [30,30,30,15,10,5] - Focus on key battles
- Adaptive: Respond to opponent's patterns

{history_str}

How do you allocate your 120 soldiers?
Respond with ONLY 6 numbers in brackets summing to 120.
Example: [25,25,25,25,10,10]

Your allocation:"""
        
        return base_prompt


class EnhancedGameEvaluator:
    """Enhanced evaluation system with comprehensive metrics"""
    
    @staticmethod
    def evaluate_experiment(game_type: str, rounds_data: List[Dict]) -> Dict[str, float]:
        """Evaluate a complete experiment"""
        
        if not rounds_data:
            return {}
        
        # Calculate rewards
        rounds_with_rewards = EnhancedGameEvaluator._calculate_rewards(game_type, rounds_data)
        
        # Calculate comprehensive metrics
        metrics = {}
        
        # Basic performance metrics
        p1_scores = [r['p1_reward'] for r in rounds_with_rewards]
        p2_scores = [r['p2_reward'] for r in rounds_with_rewards]
        
        metrics['total_score_p1'] = sum(p1_scores)
        metrics['total_score_p2'] = sum(p2_scores)
        metrics['avg_score_p1'] = np.mean(p1_scores)
        metrics['avg_score_p2'] = np.mean(p2_scores)
        
        # Game-specific metrics
        if game_type == 'prisoners_dilemma':
            metrics.update(EnhancedGameEvaluator._pd_metrics(rounds_with_rewards))
        elif game_type == 'battle_of_sexes':
            metrics.update(EnhancedGameEvaluator._bos_metrics(rounds_with_rewards))
        elif game_type == 'colonel_blotto':
            metrics.update(EnhancedGameEvaluator._blotto_metrics(rounds_with_rewards))
        
        return metrics
    
    @staticmethod
    def _calculate_rewards(game_type: str, rounds_data: List[Dict]) -> List[Dict]:
        """Calculate rewards based on actions"""
        
        for round_data in rounds_data:
            if game_type == 'prisoners_dilemma':
                payoffs = {
                    (0, 0): (3, 3), (0, 1): (0, 5),
                    (1, 0): (5, 0), (1, 1): (1, 1)
                }
                p1_reward, p2_reward = payoffs[(round_data['p1_action'], round_data['p2_action'])]
                
            elif game_type == 'battle_of_sexes':
                payoffs = {
                    (0, 0): (2, 1), (0, 1): (0, 0),
                    (1, 0): (0, 0), (1, 1): (1, 2)
                }
                p1_reward, p2_reward = payoffs[(round_data['p1_action'], round_data['p2_action'])]
                
            elif game_type == 'colonel_blotto':
                alloc1, alloc2 = round_data['p1_action'], round_data['p2_action']
                p1_wins = sum(1 for i in range(6) if alloc1[i] > alloc2[i])
                p2_wins = sum(1 for i in range(6) if alloc1[i] < alloc2[i])
                
                if p1_wins > p2_wins:
                    p1_reward, p2_reward = 1, -1
                elif p1_wins < p2_wins:
                    p1_reward, p2_reward = -1, 1
                else:
                    p1_reward, p2_reward = 0, 0
            
            round_data['p1_reward'] = p1_reward
            round_data['p2_reward'] = p2_reward
        
        return rounds_data
    
    @staticmethod
    def _pd_metrics(rounds_data: List[Dict]) -> Dict[str, float]:
        """Prisoner's Dilemma specific metrics"""
        total_rounds = len(rounds_data)
        
        # Cooperation metrics
        cooperations = sum(1 for r in rounds_data if r['p1_action'] == 0 and r['p2_action'] == 0)
        p1_cooperations = sum(1 for r in rounds_data if r['p1_action'] == 0)
        
        # Exploitation metrics
        p1_exploited = sum(1 for r in rounds_data if r['p1_action'] == 0 and r['p2_action'] == 1)
        p1_exploiting = sum(1 for r in rounds_data if r['p1_action'] == 1 and r['p2_action'] == 0)
        
        return {
            'cooperation_rate': cooperations / total_rounds,
            'p1_cooperation_rate': p1_cooperations / total_rounds,
            'p1_exploitation_rate': p1_exploited / total_rounds,
            'p1_exploiting_rate': p1_exploiting / total_rounds,
            'nash_deviation': 1 - (sum(1 for r in rounds_data if r['p1_action'] == 1) / total_rounds)
        }
    
    @staticmethod
    def _bos_metrics(rounds_data: List[Dict]) -> Dict[str, float]:
        """Battle of Sexes specific metrics"""
        total_rounds = len(rounds_data)
        
        coordinations = sum(1 for r in rounds_data if r['p1_action'] == r['p2_action'])
        p1_preference_wins = sum(1 for r in rounds_data if r['p1_action'] == 0 and r['p2_action'] == 0)
        p2_preference_wins = sum(1 for r in rounds_data if r['p1_action'] == 1 and r['p2_action'] == 1)
        
        return {
            'coordination_rate': coordinations / total_rounds,
            'p1_preference_success': p1_preference_wins / total_rounds,
            'p2_preference_success': p2_preference_wins / total_rounds,
            'fairness_balance': abs(p1_preference_wins - p2_preference_wins) / total_rounds
        }
    
    @staticmethod
    def _blotto_metrics(rounds_data: List[Dict]) -> Dict[str, float]:
        """Colonel Blotto specific metrics"""
        total_rounds = len(rounds_data)
        
        p1_wins = sum(1 for r in rounds_data if r['p1_reward'] > 0)
        ties = sum(1 for r in rounds_data if r['p1_reward'] == 0)
        
        # Calculate allocation diversity (entropy)
        entropies = []
        for r in rounds_data:
            alloc = np.array(r['p1_action'])
            prob = alloc / alloc.sum() if alloc.sum() > 0 else np.ones(6) / 6
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            entropies.append(entropy)
        
        return {
            'win_rate': p1_wins / total_rounds,
            'tie_rate': ties / total_rounds,
            'avg_allocation_entropy': np.mean(entropies),
            'allocation_consistency': 1 - np.std(entropies) / (np.mean(entropies) + 1e-6)
        }


class EnhancedVisualizationEngine:
    """Enhanced visualization engine for comprehensive analysis"""
    
    @staticmethod
    def create_experiment_visualizations(experiments: List[ExperimentResult], 
                                       output_dir: str) -> List[str]:
        """Create comprehensive visualizations"""
        
        if plt is None:
            print("⚠️  Matplotlib not available - skipping visualizations")
            return []
        
        created_files = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Model Comparison Dashboard
        dashboard_file = EnhancedVisualizationEngine._create_model_comparison_dashboard(
            experiments, output_dir, timestamp
        )
        if dashboard_file:
            created_files.append(dashboard_file)
        
        # 2. Game-specific Analysis
        for game_type in ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']:
            game_experiments = [exp for exp in experiments if exp.game_type == game_type]
            if game_experiments:
                game_file = EnhancedVisualizationEngine._create_game_analysis(
                    game_experiments, game_type, output_dir, timestamp
                )
                if game_file:
                    created_files.append(game_file)
        
        # 3. Performance Evolution Charts
        evolution_file = EnhancedVisualizationEngine._create_performance_evolution(
            experiments, output_dir, timestamp
        )
        if evolution_file:
            created_files.append(evolution_file)
        
        return created_files
    
    @staticmethod
    def _create_model_comparison_dashboard(experiments: List[ExperimentResult], 
                                         output_dir: str, timestamp: str) -> Optional[str]:
        """Create model comparison dashboard"""
        
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Organize data by model and game
            model_game_metrics = {}
            for exp in experiments:
                if not exp.success:
                    continue
                    
                model_name = exp.model_config.model_name
                game_type = exp.game_type
                
                if model_name not in model_game_metrics:
                    model_game_metrics[model_name] = {}
                
                model_game_metrics[model_name][game_type] = exp.metrics
            
            if not model_game_metrics:
                return None
            
            # Create subplots
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
            
            # 1. Overall Performance Heatmap
            ax1 = fig.add_subplot(gs[0, :2])
            models = list(model_game_metrics.keys())
            games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
            
            performance_matrix = np.zeros((len(models), len(games)))
            for i, model in enumerate(models):
                for j, game in enumerate(games):
                    if game in model_game_metrics[model]:
                        performance_matrix[i, j] = model_game_metrics[model][game].get('avg_score_p1', 0)
            
            im = ax1.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
            ax1.set_xticks(range(len(games)))
            ax1.set_xticklabels([g.replace('_', ' ').title() for g in games])
            ax1.set_yticks(range(len(models)))
            ax1.set_yticklabels(models, rotation=45, ha='right')
            ax1.set_title('Average Score by Model and Game', fontsize=14, fontweight='bold')
            
            # Add value annotations
            for i in range(len(models)):
                for j in range(len(games)):
                    text = ax1.text(j, i, f'{performance_matrix[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax1)
            
            # 2. Cooperation Rate Comparison (for PD)
            ax2 = fig.add_subplot(gs[0, 2])
            pd_coop_rates = []
            pd_models = []
            for model, games_data in model_game_metrics.items():
                if 'prisoners_dilemma' in games_data:
                    pd_coop_rates.append(games_data['prisoners_dilemma'].get('cooperation_rate', 0))
                    pd_models.append(model)
            
            if pd_coop_rates:
                bars = ax2.bar(range(len(pd_models)), pd_coop_rates, 
                              color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(pd_models)])
                ax2.set_xticks(range(len(pd_models)))
                ax2.set_xticklabels(pd_models, rotation=45, ha='right')
                ax2.set_ylabel('Cooperation Rate')
                ax2.set_title('Cooperation in Prisoner\'s Dilemma', fontweight='bold')
                ax2.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, rate in zip(bars, pd_coop_rates):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{rate:.2f}', ha='center', va='bottom')
            
            # 3-5. Game-specific performance charts
            for idx, game_type in enumerate(games):
                ax = fig.add_subplot(gs[1 + idx//2, idx%2])
                
                game_models = []
                game_scores = []
                
                for model, games_data in model_game_metrics.items():
                    if game_type in games_data:
                        game_models.append(model)
                        game_scores.append(games_data[game_type].get('avg_score_p1', 0))
                
                if game_scores:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(game_models)))
                    bars = ax.bar(range(len(game_models)), game_scores, color=colors)
                    ax.set_xticks(range(len(game_models)))
                    ax.set_xticklabels(game_models, rotation=45, ha='right')
                    ax.set_ylabel('Average Score')
                    ax.set_title(f'{game_type.replace("_", " ").title()} Performance', fontweight='bold')
                    
                    # Add value labels
                    for bar, score in zip(bars, game_scores):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = f"{output_dir}/model_comparison_dashboard_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Model comparison dashboard saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating dashboard: {e}")
            return None
    
    @staticmethod
    def _create_game_analysis(experiments: List[ExperimentResult], game_type: str,
                            output_dir: str, timestamp: str) -> Optional[str]:
        """Create detailed game-specific analysis"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{game_type.replace("_", " ").title()} Detailed Analysis', 
                        fontsize=16, fontweight='bold')
            
            # Collect data across all experiments for this game
            all_rounds_data = []
            model_names = []
            
            for exp in experiments:
                if exp.game_type == game_type and exp.success:
                    all_rounds_data.extend(exp.rounds_data)
                    model_names.append(exp.model_config.model_name)
            
            if not all_rounds_data:
                return None
            
            # 1. Score evolution over rounds
            ax1 = axes[0, 0]
            for exp in experiments:
                if exp.game_type == game_type and exp.success:
                    rounds = exp.rounds_data
                    scores = [r.get('p1_reward', 0) for r in rounds]
                    cumulative_scores = np.cumsum(scores)
                    ax1.plot(range(1, len(cumulative_scores) + 1), cumulative_scores, 
                            label=exp.model_config.model_name, linewidth=2)
            
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Cumulative Score')
            ax1.set_title('Cumulative Score Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Action distribution
            ax2 = axes[0, 1]
            if game_type in ['prisoners_dilemma', 'battle_of_sexes']:
                actions = [r.get('p1_action', 0) for r in all_rounds_data]
                action_counts = [actions.count(0), actions.count(1)]
                labels = ['Cooperate/Preference 0', 'Defect/Preference 1']
                ax2.pie(action_counts, labels=labels, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Action Distribution')
            
            # 3. Game-specific metric
            ax3 = axes[1, 0]
            if game_type == 'prisoners_dilemma':
                # Cooperation rate over time
                cooperation_rates = []
                window_size = 5
                for i in range(len(all_rounds_data) - window_size + 1):
                    window = all_rounds_data[i:i+window_size]
                    cooperations = sum(1 for r in window if r.get('p1_action') == 0 and r.get('p2_action') == 0)
                    cooperation_rates.append(cooperations / window_size)
                
                ax3.plot(range(window_size, len(all_rounds_data) + 1), cooperation_rates, 'g-', linewidth=2)
                ax3.set_xlabel('Round')
                ax3.set_ylabel('Cooperation Rate (5-round window)')
                ax3.set_title('Cooperation Trend')
                ax3.grid(True, alpha=0.3)
            
            # 4. Model performance comparison
            ax4 = axes[1, 1]
            model_performance = {}
            for exp in experiments:
                if exp.game_type == game_type and exp.success:
                    model_name = exp.model_config.model_name
                    avg_score = exp.metrics.get('avg_score_p1', 0)
                    model_performance[model_name] = avg_score
            
            if model_performance:
                models = list(model_performance.keys())
                scores = list(model_performance.values())
                colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
                
                bars = ax4.bar(models, scores, color=colors)
                ax4.set_ylabel('Average Score')
                ax4.set_title('Model Performance Comparison')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = f"{output_dir}/{game_type}_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 {game_type} analysis saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating game analysis: {e}")
            return None
    
    @staticmethod
    def _create_performance_evolution(experiments: List[ExperimentResult],
                                    output_dir: str, timestamp: str) -> Optional[str]:
        """Create performance evolution visualization"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Model Performance Evolution Analysis', fontsize=16, fontweight='bold')
            
            # Group experiments by model
            model_experiments = {}
            for exp in experiments:
                if exp.success:
                    model_name = exp.model_config.model_name
                    if model_name not in model_experiments:
                        model_experiments[model_name] = []
                    model_experiments[model_name].append(exp)
            
            if not model_experiments:
                return None
            
            # 1. Average performance across games
            ax1 = axes[0, 0]
            models = list(model_experiments.keys())
            avg_performances = []
            
            for model in models:
                model_scores = []
                for exp in model_experiments[model]:
                    model_scores.append(exp.metrics.get('avg_score_p1', 0))
                avg_performances.append(np.mean(model_scores) if model_scores else 0)
            
            bars = ax1.bar(models, avg_performances, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
            ax1.set_ylabel('Average Score Across Games')
            ax1.set_title('Overall Model Performance')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, perf in zip(bars, avg_performances):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{perf:.2f}', ha='center', va='bottom')
            
            # 2. Performance variance
            ax2 = axes[0, 1]
            performance_vars = []
            for model in models:
                model_scores = []
                for exp in model_experiments[model]:
                    model_scores.append(exp.metrics.get('avg_score_p1', 0))
                performance_vars.append(np.std(model_scores) if len(model_scores) > 1 else 0)
            
            bars = ax2.bar(models, performance_vars, color='lightcoral')
            ax2.set_ylabel('Performance Standard Deviation')
            ax2.set_title('Performance Consistency')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Game-specific performance radar chart (simplified)
            ax3 = axes[1, 0]
            games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
            
            # Create a simple line plot instead of radar for clarity
            for i, model in enumerate(models):
                game_scores = []
                for game in games:
                    game_exp = [exp for exp in model_experiments[model] if exp.game_type == game]
                    if game_exp:
                        game_scores.append(game_exp[0].metrics.get('avg_score_p1', 0))
                    else:
                        game_scores.append(0)
                
                ax3.plot(games, game_scores, marker='o', linewidth=2, label=model)
            
            ax3.set_ylabel('Average Score')
            ax3.set_title('Performance by Game Type')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 4. Success rate and completion analysis
            ax4 = axes[1, 1]
            total_experiments = {}
            successful_experiments = {}
            
            for exp in experiments:
                model_name = exp.model_config.model_name
                total_experiments[model_name] = total_experiments.get(model_name, 0) + 1
                if exp.success:
                    successful_experiments[model_name] = successful_experiments.get(model_name, 0) + 1
            
            success_rates = []
            for model in models:
                total = total_experiments.get(model, 1)
                successful = successful_experiments.get(model, 0)
                success_rates.append(successful / total)
            
            bars = ax4.bar(models, success_rates, color='lightgreen')
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Experiment Success Rate')
            ax4.set_ylim(0, 1)
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = f"{output_dir}/performance_evolution_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Performance evolution saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Error creating performance evolution: {e}")
            return None


class MultiAgentExperimentRunner:
    """Main experiment runner supporting multiple models and comprehensive analysis"""
    
    def __init__(self):
        self.experiments: List[ExperimentResult] = []
        self.output_dir = "../../experiments/enhanced_multi_agent"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_model_config(self, provider: str, api_key: str, model_name: str, **kwargs) -> ModelConfig:
        """Add a new model configuration"""
        return ModelConfig(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )
    
    def run_experiment(self, model_config: ModelConfig, game_type: str, 
                      num_rounds: int = 10, prompt_variant: str = "standard") -> ExperimentResult:
        """Run a single experiment"""
        
        experiment_id = f"{model_config.model_name}_{game_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n🎮 Starting experiment: {experiment_id}")
        print(f"Model: {model_config.model_name} ({model_config.provider})")
        print(f"Game: {game_type}")
        print(f"Rounds: {num_rounds}")
        
        try:
            # Initialize API client
            api_client = EnhancedAPIClient(model_config)
            
            # Run the game
            rounds_data = []
            for round_num in range(num_rounds):
                print(f"  Round {round_num + 1}/{num_rounds}", end=" ")
                
                # Get history for this round
                history = [(r['p1_action'], r['p2_action']) for r in rounds_data]
                
                # Generate prompt
                prompt = GamePromptGenerator.get_prompt(
                    game_type, round_num, history, prompt_variant, model_config.model_name
                )
                
                # Get LLM response
                try:
                    llm_response = api_client.call_model(prompt)
                    player_action = self._parse_response(llm_response, game_type)
                    
                    # Get baseline opponent action
                    opponent_action = self._get_baseline_action(game_type, history)
                    
                    # Record round
                    round_data = {
                        'round': round_num,
                        'p1_action': player_action,
                        'p2_action': opponent_action,
                        'llm_response': llm_response[:200]  # Truncate for storage
                    }
                    rounds_data.append(round_data)
                    print(f"✅ P1={player_action}, P2={opponent_action}")
                    
                    # Brief pause to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    # Continue with default action
                    default_action = 0 if game_type != 'colonel_blotto' else [20, 20, 20, 20, 20, 20]
                    round_data = {
                        'round': round_num,
                        'p1_action': default_action,
                        'p2_action': self._get_baseline_action(game_type, []),
                        'llm_response': f"ERROR: {str(e)}"
                    }
                    rounds_data.append(round_data)
            
            # Evaluate experiment
            metrics = EnhancedGameEvaluator.evaluate_experiment(game_type, rounds_data)
            
            # Create experiment result
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                model_config=model_config,
                game_type=game_type,
                num_rounds=num_rounds,
                rounds_data=rounds_data,
                metrics=metrics,
                timestamp=datetime.datetime.now().isoformat(),
                success=True
            )
            
            self.experiments.append(experiment_result)
            
            print(f"✅ Experiment completed successfully")
            print(f"   Final score: {metrics.get('total_score_p1', 0):.1f}")
            print(f"   Key metrics: {self._format_key_metrics(game_type, metrics)}")
            
            return experiment_result
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            
            # Create failed experiment result
            failed_result = ExperimentResult(
                experiment_id=experiment_id,
                model_config=model_config,
                game_type=game_type,
                num_rounds=num_rounds,
                rounds_data=[],
                metrics={},
                timestamp=datetime.datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
            
            self.experiments.append(failed_result)
            return failed_result
    
    def _parse_response(self, response: str, game_type: str) -> Any:
        """Parse LLM response into game action"""
        
        if game_type == 'colonel_blotto':
            # Parse list format [x,x,x,x,x,x]
            match = re.search(r'\[([^\]]+)\]', response)
            if match:
                try:
                    numbers = [int(float(x.strip())) for x in match.group(1).split(',')]
                    if len(numbers) == 6 and sum(numbers) == 120:
                        return numbers
                    else:
                        # Normalize to sum to 120
                        total = sum(numbers) if sum(numbers) > 0 else 120
                        normalized = [int(n * 120 / total) for n in numbers]
                        # Adjust for rounding errors
                        diff = 120 - sum(normalized)
                        normalized[0] += diff
                        return normalized[:6] + [0] * (6 - len(normalized))
                except:
                    pass
            
            # Default uniform allocation
            return [20, 20, 20, 20, 20, 20]
        
        else:
            # Parse single number (0 or 1)
            numbers = re.findall(r'\b[01]\b', response)
            if numbers:
                return int(numbers[0])
            
            # Try to extract any number and see if it's 0 or 1
            all_numbers = re.findall(r'\d+', response)
            valid_actions = [int(n) for n in all_numbers if int(n) in [0, 1]]
            if valid_actions:
                return valid_actions[0]
            
            # Default to cooperate
            return 0
    
    def _get_baseline_action(self, game_type: str, history: List[Tuple]) -> Any:
        """Get baseline opponent action"""
        
        if game_type == 'prisoners_dilemma':
            # Tit-for-tat baseline
            if not history:
                return 0  # Start cooperative
            return history[-1][0]  # Copy player's last action
        
        elif game_type == 'battle_of_sexes':
            # Alternating baseline
            return len(history) % 2
        
        elif game_type == 'colonel_blotto':
            # Uniform distribution baseline
            return [20, 20, 20, 20, 20, 20]
    
    def _format_key_metrics(self, game_type: str, metrics: Dict[str, float]) -> str:
        """Format key metrics for display"""
        
        if game_type == 'prisoners_dilemma':
            coop_rate = metrics.get('cooperation_rate', 0)
            return f"Cooperation: {coop_rate:.1%}"
        elif game_type == 'battle_of_sexes':
            coord_rate = metrics.get('coordination_rate', 0)
            return f"Coordination: {coord_rate:.1%}"
        elif game_type == 'colonel_blotto':
            win_rate = metrics.get('win_rate', 0)
            return f"Win Rate: {win_rate:.1%}"
        else:
            return "N/A"
    
    def run_model_comparison_study(self, model_configs: List[ModelConfig], 
                                 games: List[str] = None, num_rounds: int = 10) -> Dict[str, Any]:
        """Run comprehensive model comparison study"""
        
        if games is None:
            games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
        
        print(f"\n🚀 MULTI-AGENT MODEL COMPARISON STUDY")
        print(f"Models: {[config.model_name for config in model_configs]}")
        print(f"Games: {games}")
        print(f"Rounds per game: {num_rounds}")
        print("=" * 60)
        
        # Run all experiments
        for model_config in model_configs:
            for game_type in games:
                self.run_experiment(model_config, game_type, num_rounds)
        
        # Generate comprehensive analysis
        results = self.generate_comprehensive_analysis()
        
        print(f"\n🎉 Study completed! Results saved to: {self.output_dir}")
        return results
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of all experiments"""
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw experiment data
        self._save_raw_data(timestamp)
        
        # Generate visualizations
        viz_files = EnhancedVisualizationEngine.create_experiment_visualizations(
            self.experiments, self.output_dir
        )
        
        # Generate summary report
        summary = self._generate_summary_report(timestamp)
        
        # Generate MLGym format results
        mlgym_results = self._generate_mlgym_results()
        
        return {
            'summary': summary,
            'mlgym_results': mlgym_results,
            'visualization_files': viz_files,
            'total_experiments': len(self.experiments),
            'successful_experiments': len([exp for exp in self.experiments if exp.success]),
            'timestamp': timestamp
        }
    
    def _save_raw_data(self, timestamp: str) -> str:
        """Save raw experiment data"""
        
        # Convert experiments to serializable format
        serializable_experiments = []
        for exp in self.experiments:
            exp_dict = {
                'experiment_id': exp.experiment_id,
                'model_config': {
                    'provider': exp.model_config.provider,
                    'model_name': exp.model_config.model_name,
                    'temperature': exp.model_config.temperature
                },
                'game_type': exp.game_type,
                'num_rounds': exp.num_rounds,
                'rounds_data': exp.rounds_data,
                'metrics': exp.metrics,
                'timestamp': exp.timestamp,
                'success': exp.success,
                'error_message': exp.error_message
            }
            serializable_experiments.append(exp_dict)
        
        filename = f"{self.output_dir}/raw_experiments_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(serializable_experiments, f, indent=2, default=str)
        
        print(f"💾 Raw data saved: {filename}")
        return filename
    
    def _generate_summary_report(self, timestamp: str) -> Dict[str, Any]:
        """Generate summary report"""
        
        successful_experiments = [exp for exp in self.experiments if exp.success]
        
        # Group by model and game
        model_performance = {}
        game_performance = {}
        
        for exp in successful_experiments:
            model_name = exp.model_config.model_name
            game_type = exp.game_type
            
            # Model performance
            if model_name not in model_performance:
                model_performance[model_name] = {
                    'experiments': 0,
                    'total_score': 0,
                    'games': {}
                }
            
            model_performance[model_name]['experiments'] += 1
            model_performance[model_name]['total_score'] += exp.metrics.get('avg_score_p1', 0)
            model_performance[model_name]['games'][game_type] = exp.metrics
            
            # Game performance
            if game_type not in game_performance:
                game_performance[game_type] = {
                    'experiments': 0,
                    'models': {}
                }
            
            game_performance[game_type]['experiments'] += 1
            game_performance[game_type]['models'][model_name] = exp.metrics
        
        # Calculate averages
        for model_name in model_performance:
            exp_count = model_performance[model_name]['experiments']
            model_performance[model_name]['avg_score'] = (
                model_performance[model_name]['total_score'] / exp_count
            )
        
        summary = {
            'model_performance': model_performance,
            'game_performance': game_performance,
            'total_experiments': len(self.experiments),
            'successful_experiments': len(successful_experiments),
            'success_rate': len(successful_experiments) / len(self.experiments) if self.experiments else 0
        }
        
        # Save summary
        filename = f"{self.output_dir}/summary_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Summary report saved: {filename}")
        
        # Print console summary
        self._print_console_summary(summary)
        
        return summary
    
    def _print_console_summary(self, summary: Dict[str, Any]):
        """Print summary to console"""
        
        print(f"\n" + "="*80)
        print("🎮 MULTI-AGENT GAME THEORY STUDY SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Experiments: {summary['total_experiments']}")
        print(f"   Successful: {summary['successful_experiments']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        for model_name, perf in summary['model_performance'].items():
            print(f"   {model_name}:")
            print(f"      Average Score: {perf['avg_score']:.2f}")
            print(f"      Experiments: {perf['experiments']}")
            
            for game_type, metrics in perf['games'].items():
                key_metric = self._get_key_metric_for_game(game_type, metrics)
                print(f"      {game_type}: {key_metric}")
        
        print(f"\n🎯 GAME TYPE ANALYSIS:")
        for game_type, perf in summary['game_performance'].items():
            print(f"   {game_type.replace('_', ' ').title()}:")
            print(f"      Experiments: {perf['experiments']}")
            
            # Find best performing model
            best_model = None
            best_score = -float('inf')
            for model_name, metrics in perf['models'].items():
                score = metrics.get('avg_score_p1', 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model:
                print(f"      Best Model: {best_model} (Score: {best_score:.2f})")
        
        print("="*80)
    
    def _get_key_metric_for_game(self, game_type: str, metrics: Dict[str, float]) -> str:
        """Get key metric string for game type"""
        
        if game_type == 'prisoners_dilemma':
            score = metrics.get('avg_score_p1', 0)
            coop = metrics.get('cooperation_rate', 0)
            return f"Score: {score:.1f}, Cooperation: {coop:.1%}"
        elif game_type == 'battle_of_sexes':
            score = metrics.get('avg_score_p1', 0)
            coord = metrics.get('coordination_rate', 0)
            return f"Score: {score:.1f}, Coordination: {coord:.1%}"
        elif game_type == 'colonel_blotto':
            score = metrics.get('avg_score_p1', 0)
            win_rate = metrics.get('win_rate', 0)
            return f"Score: {score:.1f}, Win Rate: {win_rate:.1%}"
        else:
            return f"Score: {metrics.get('avg_score_p1', 0):.1f}"
    
    def _generate_mlgym_results(self) -> Dict[str, float]:
        """Generate results in MLGym format"""
        
        successful_experiments = [exp for exp in self.experiments if exp.success]
        
        if not successful_experiments:
            return {}
        
        # Calculate overall metrics
        all_scores = [exp.metrics.get('avg_score_p1', 0) for exp in successful_experiments]
        overall_score = np.mean(all_scores)
        
        # Game-specific metrics
        mlgym_results = {
            'Overall_Score': overall_score,
            'Total_Experiments': len(successful_experiments)
        }
        
        # Add game-specific results
        games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
        game_abbrev = {'prisoners_dilemma': 'PD', 'battle_of_sexes': 'BoS', 'colonel_blotto': 'Blotto'}
        
        for game_type in games:
            game_experiments = [exp for exp in successful_experiments if exp.game_type == game_type]
            if game_experiments:
                game_scores = [exp.metrics.get('avg_score_p1', 0) for exp in game_experiments]
                abbrev = game_abbrev[game_type]
                
                mlgym_results[f'{abbrev}_Score'] = np.mean(game_scores)
                mlgym_results[f'{abbrev}_Experiments'] = len(game_experiments)
                
                # Add game-specific metrics
                if game_type == 'prisoners_dilemma':
                    coop_rates = [exp.metrics.get('cooperation_rate', 0) for exp in game_experiments]
                    mlgym_results[f'{abbrev}_Cooperation_Rate'] = np.mean(coop_rates)
                elif game_type == 'battle_of_sexes':
                    coord_rates = [exp.metrics.get('coordination_rate', 0) for exp in game_experiments]
                    mlgym_results[f'{abbrev}_Coordination_Rate'] = np.mean(coord_rates)
                elif game_type == 'colonel_blotto':
                    win_rates = [exp.metrics.get('win_rate', 0) for exp in game_experiments]
                    mlgym_results[f'{abbrev}_Win_Rate'] = np.mean(win_rates)
        
        return mlgym_results


# Utility functions for easy setup
def create_model_configs() -> List[ModelConfig]:
    """Create model configurations interactively"""
    
    configs = []
    
    print("🤖 MODEL CONFIGURATION SETUP")
    print("Available providers: openai, together, anthropic")
    
    while True:
        print(f"\nModel {len(configs) + 1}:")
        provider = input("Provider (openai/together/anthropic or 'done' to finish): ").lower()
        
        if provider == 'done':
            break
        
        if provider not in ['openai', 'together', 'anthropic']:
            print("Invalid provider. Please choose: openai, together, anthropic")
            continue
        
        api_key = input(f"API key for {provider}: ")
        
        # Suggest models based on provider
        if provider == 'openai':
            print("Suggested models: gpt-4, gpt-3.5-turbo")
            model_name = input("Model name: ") or "gpt-4"
        elif provider == 'together':
            print("Suggested models:")
            print("  - meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
            print("  - deepseek-ai/DeepSeek-R1-0528")
            print("  - meta-llama/Llama-2-70b-chat-hf")
            model_name = input("Model name: ") or "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        elif provider == 'anthropic':
            print("Suggested models: claude-3-opus-20240229, claude-3-sonnet-20240229")
            model_name = input("Model name: ") or "claude-3-sonnet-20240229"
        
        config = ModelConfig(
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )
        
        configs.append(config)
        print(f"✅ Added {model_name} ({provider})")
    
    return configs


def run_quick_comparison():
    """Run a quick model comparison study"""
    
    print("🚀 QUICK MODEL COMPARISON STUDY")
    
    # Get model configurations
    model_configs = create_model_configs()
    
    if not model_configs:
        print("No models configured. Exiting.")
        return
    
    # Setup experiment runner
    runner = MultiAgentExperimentRunner()
    
    # Run study
    games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
    num_rounds = int(input("Number of rounds per game (default 10): ") or "10")
    
    results = runner.run_model_comparison_study(model_configs, games, num_rounds)
    
    print(f"\n🎯 MLGYM FORMAT RESULTS:")
    print("="*40)
    for metric, value in results['mlgym_results'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")
    
    return results


if __name__ == "__main__":
    print("🎮 ENHANCED MULTI-AGENT GAME THEORY SYSTEM")
    print("1. Run quick model comparison")
    print("2. Setup custom experiment")
    
    choice = input("Choose option (1/2): ")
    
    if choice == "1":
        run_quick_comparison()
    elif choice == "2":
        print("Custom experiment setup not implemented yet.")
        print("Use run_quick_comparison() for now.")
    else:
        print("Invalid choice")