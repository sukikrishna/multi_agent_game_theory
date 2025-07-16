"""
Automated API-based Data Collection for Game Theory Experiments
Supports OpenAI GPT models and other API-compatible services
"""

import json
import datetime
import os
import re
import time
from typing import List, Dict, Any, Tuple
import requests
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for different API providers"""
    provider: str
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 150
    temperature: float = 0.7


class AutomatedDataCollector:
    """Collect LLM responses automatically via API"""
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, game_type: str, 
                        player1_model: str, num_rounds: int = 10):
        """Start a new automated experiment"""
        self.current_experiment = {
            'experiment_name': experiment_name,
            'game_type': game_type,
            'player1_model': player1_model,
            'player2_model': 'baseline',
            'num_rounds': num_rounds,
            'timestamp': datetime.datetime.now().isoformat(),
            'rounds': [],
            'completed': False,
            'api_config': {
                'provider': self.api_config.provider,
                'model': self.api_config.model_name,
                'temperature': self.api_config.temperature
            }
        }
        print(f"🎮 Started automated experiment: {experiment_name}")
        print(f"📊 Game: {game_type}, Model: {player1_model}, Rounds: {num_rounds}")
    
    def call_llm_api(self, prompt: str) -> str:
        """Make API call to LLM provider"""
        
        if self.api_config.provider.lower() == 'openai':
            return self._call_openai_api(prompt)
        elif self.api_config.provider.lower() == 'together':
            return self._call_together_api(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.api_config.provider}")
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.api_config.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.api_config.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
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
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
    
    def _call_together_api(self, prompt: str) -> str:
        """Call Together AI API"""
        headers = {
            'Authorization': f'Bearer {self.api_config.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.api_config.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
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
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
    
    def get_game_prompt(self, game_type: str, round_num: int, history: List[Tuple]) -> str:
        """Generate the prompt for LLM"""
        
        if game_type == "prisoners_dilemma":
            return self._get_pd_prompt(round_num, history)
        elif game_type == "battle_of_sexes":
            return self._get_bos_prompt(round_num, history)
        elif game_type == "colonel_blotto":
            return self._get_blotto_prompt(round_num, history)
        else:
            raise ValueError(f"Unknown game type: {game_type}")
    
    def _get_pd_prompt(self, round_num: int, history: List[Tuple]) -> str:
        """Prisoner's Dilemma prompt"""
        history_str = "\n".join([f"Round {i+1}: You played {h[0]}, Opponent played {h[1]}" 
                                for i, h in enumerate(history)])
        
        prompt = f"""PRISONER'S DILEMMA - Round {round_num + 1}

You are playing an iterated Prisoner's Dilemma game. 

RULES:
- Choose 0 (Cooperate) or 1 (Defect)
- Payoff Matrix (Your score, Opponent score):
  • Both cooperate (0,0): (3,3) - Good for both
  • You cooperate, they defect (0,1): (0,5) - You get exploited  
  • You defect, they cooperate (1,0): (5,0) - You exploit them
  • Both defect (1,1): (1,1) - Bad for both

GAME HISTORY:
{history_str if history else "This is the first round."}

What is your choice for this round?
Respond with ONLY the number: 0 or 1

Your choice:"""
        return prompt
    
    def _get_bos_prompt(self, round_num: int, history: List[Tuple]) -> str:
        """Battle of Sexes prompt"""
        history_str = "\n".join([f"Round {i+1}: You chose {h[0]}, Opponent chose {h[1]}" 
                                for i, h in enumerate(history)])
        
        prompt = f"""BATTLE OF SEXES - Round {round_num + 1}

You are playing a coordination game with conflicting preferences.

RULES:
- Choose 0 (Your preference) or 1 (Opponent's preference)
- Payoff Matrix (Your score, Opponent score):
  • Both choose 0: (2,1) - You get your preference
  • Both choose 1: (1,2) - Opponent gets their preference
  • Different choices: (0,0) - Nobody gets anything

GOAL: Coordinate on the same choice, preferably yours!

GAME HISTORY:
{history_str if history else "This is the first round."}

What is your choice for this round?
Respond with ONLY the number: 0 or 1

Your choice:"""
        return prompt
    
    def _get_blotto_prompt(self, round_num: int, history: List[Tuple]) -> str:
        """Colonel Blotto prompt"""
        if history:
            last_round = history[-1]
            history_str = f"""
LAST ROUND EXAMPLE:
Your allocation: {last_round[0]}
Opponent allocation: {last_round[1]}

RECENT HISTORY:
""" + "\n".join([f"Round {i+1}: You: {h[0]}, Opponent: {h[1]}" 
                 for i, h in enumerate(history[-3:])])
        else:
            history_str = "This is the first round."
        
        prompt = f"""COLONEL BLOTTO - Round {round_num + 1}

You are commanding an army in the Colonel Blotto game.

RULES:
- Allocate exactly 120 soldiers across 6 battlefields
- Win a battlefield if you have MORE soldiers there than your opponent
- Win the game if you win MORE battlefields than your opponent
- Scoring: +1 for winning, -1 for losing, 0 for tie

STRATEGY TIPS:
- Spread evenly: [20,20,20,20,20,20]
- Concentrate: [40,40,40,0,0,0] 
- Counter opponent's weak spots

{history_str}

Allocate your 120 soldiers across 6 battlefields.
Respond with ONLY 6 numbers in brackets that sum to 120.
Example: [25,25,25,25,10,10]

Your allocation:"""
        return prompt
    
    def parse_response(self, llm_response: str, game_type: str) -> Any:
        """Parse LLM response into action"""
        
        if game_type == 'colonel_blotto':
            # Parse list format [x,x,x,x,x,x]
            match = re.search(r'\[([^\]]+)\]', llm_response)
            if match:
                try:
                    numbers = [int(x.strip()) for x in match.group(1).split(',')]
                    if len(numbers) == 6 and sum(numbers) == 120:
                        return numbers
                    else:
                        print(f"⚠️  Invalid Blotto allocation: {numbers} (sum: {sum(numbers)})")
                        return [20, 20, 20, 20, 20, 20]  # Default uniform
                except ValueError:
                    print(f"⚠️  Could not parse numbers in: {llm_response}")
                    return [20, 20, 20, 20, 20, 20]
            else:
                print(f"⚠️  No valid allocation found in: {llm_response}")
                return [20, 20, 20, 20, 20, 20]
        else:
            # Parse single number (0 or 1)
            numbers = re.findall(r'\b[01]\b', llm_response)
            if numbers:
                return int(numbers[0])
            else:
                # Try to extract any number and see if it's 0 or 1
                all_numbers = re.findall(r'\d+', llm_response)
                valid_actions = [int(n) for n in all_numbers if int(n) in [0, 1]]
                if valid_actions:
                    return valid_actions[0]
                else:
                    print(f"⚠️  Could not find valid action (0 or 1) in: {llm_response}, defaulting to 0")
                    return 0  # Default to cooperate
    
    def get_baseline_action(self, game_type: str, history: List[Tuple]) -> Any:
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
    
    def run_experiment(self) -> bool:
        """Run the complete automated experiment"""
        
        if not self.current_experiment:
            print("❌ No active experiment! Call start_experiment() first.")
            return False
        
        game_type = self.current_experiment['game_type']
        num_rounds = self.current_experiment['num_rounds']
        
        print(f"\n🚀 Running {num_rounds} rounds of {game_type}...")
        
        # Run all rounds
        for round_num in range(num_rounds):
            print(f"\n🎯 Round {round_num + 1}/{num_rounds}")
            
            # Get history for prompt
            history = [(r['player_action'], r['opponent_action']) 
                      for r in self.current_experiment['rounds']]
            
            # Generate prompt
            prompt = self.get_game_prompt(game_type, round_num, history)
            
            try:
                # Get LLM response
                print("🤖 Calling LLM API...")
                llm_response = self.call_llm_api(prompt)
                
                # Parse response
                player_action = self.parse_response(llm_response, game_type)
                
                # Get opponent action
                opponent_action = self.get_baseline_action(game_type, history)
                
                # Record round
                round_data = {
                    'round': round_num,
                    'player_action': player_action,
                    'opponent_action': opponent_action,
                    'llm_response': llm_response.strip()
                }
                
                self.current_experiment['rounds'].append(round_data)
                print(f"✅ Round {round_num + 1}: You={player_action}, Opponent={opponent_action}")
                
                # Brief pause to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error in round {round_num + 1}: {e}")
                return False
        
        # Mark experiment complete
        self.current_experiment['completed'] = True
        self.current_experiment['total_rounds'] = len(self.current_experiment['rounds'])
        
        # Add to experiments list
        self.experiments.append(self.current_experiment.copy())
        
        print(f"\n🎉 Experiment completed: {self.current_experiment['experiment_name']}")
        print(f"📈 Total rounds: {self.current_experiment['total_rounds']}")
        
        return True
    
    def save_and_evaluate(self, output_dir: str = "../../experiments/api_data") -> Dict[str, str]:
        """Save data and run evaluation"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw experiment data
        raw_file = f"{output_dir}/api_experiments_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        # Convert to evaluation format
        eval_data = self.convert_to_evaluation_format()
        eval_file = f"{output_dir}/evaluation_data_{timestamp}.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"💾 Raw data saved: {raw_file}")
        print(f"📊 Evaluation data saved: {eval_file}")
        
        # Run evaluation automatically
        try:
            from simple_evaluator import evaluate_manual_data
            results = evaluate_manual_data(eval_file)
            
            # Save evaluation results
            results_file = f"{output_dir}/evaluation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"📈 Evaluation results saved: {results_file}")
            
            # Print summary
            self.print_experiment_summary(results)
            
            return {
                'raw_data': raw_file,
                'evaluation_data': eval_file,
                'results': results_file
            }
            
        except Exception as e:
            print(f"⚠️  Could not run automatic evaluation: {e}")
            return {
                'raw_data': raw_file,
                'evaluation_data': eval_file
            }
    
    def convert_to_evaluation_format(self) -> Dict:
        """Convert experiment data to evaluation format"""
        
        evaluation_data = {
            'prisoners_dilemma': [],
            'battle_of_sexes': [],
            'colonel_blotto': []
        }
        
        for exp in self.experiments:
            if not exp['completed']:
                continue
                
            game_type = exp['game_type']
            
            session_data = {
                'timestamp': exp['timestamp'],
                'player1_strategy': exp['player1_model'],
                'player2_strategy': exp['player2_model'], 
                'num_rounds': exp['total_rounds'],
                'results': []
            }
            
            for round_data in exp['rounds']:
                result = {
                    'round': round_data['round'],
                    'p1_action': round_data['player_action'],
                    'p2_action': round_data['opponent_action'],
                    'p1_reward': 0,  # Will be calculated by evaluator
                    'p2_reward': 0
                }
                session_data['results'].append(result)
            
            evaluation_data[game_type].append(session_data)
        
        return evaluation_data
    
    def print_experiment_summary(self, results: Dict):
        """Print a nice summary of results"""
        
        print("\n" + "="*60)
        print("🎮 EXPERIMENT SUMMARY")
        print("="*60)
        
        summary = results.get('summary', {})
        
        for game_type, metrics in summary.items():
            if isinstance(metrics, dict) and game_type != 'overall':
                print(f"\n🎯 {game_type.replace('_', ' ').title()}:")
                print(f"   Score: {metrics.get('avg_score', 0):.1f}")
                print(f"   Coordination Rate: {metrics.get('avg_coordination_rate', 0):.1%}")
                print(f"   Nash Deviation: {metrics.get('avg_nash_deviation', 0):.3f}")
                print(f"   Regret: {metrics.get('avg_regret', 0):.2f}")
        
        if 'overall' in summary:
            overall = summary['overall']
            print(f"\n🏆 OVERALL PERFORMANCE:")
            print(f"   Overall Score: {overall.get('overall_score', 0):.1f}")
            print(f"   Overall Coordination: {overall.get('overall_coordination_rate', 0):.1%}")
            print(f"   Overall Nash Deviation: {overall.get('overall_nash_deviation', 0):.3f}")
        
        print("="*60)


# Configuration examples for different providers
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


def run_quick_experiment():
    """Example usage - run a quick experiment"""
    
    # Set up API configuration
    api_key = input("Enter your API key: ")
    provider = input("Enter provider (openai/together): ").lower()
    
    if provider == "openai":
        config = get_openai_config(api_key, "gpt-4")
    elif provider == "together":
        model = input("Enter model name (or press enter for Llama-2-70b): ") or "meta-llama/Llama-2-70b-chat-hf"
        config = get_together_config(api_key, model)
    else:
        print("❌ Unsupported provider")
        return
    
    # Initialize collector
    collector = AutomatedDataCollector(config)
    
    # Run experiments for all three games
    games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
    
    for game in games:
        print(f"\n🎮 Starting {game} experiment...")
        collector.start_experiment(
            experiment_name=f"api_test_{game}",
            game_type=game,
            player1_model=config.model_name,
            num_rounds=10
        )
        
        success = collector.run_experiment()
        if not success:
            print(f"❌ Failed to complete {game} experiment")
            break
    
    # Save and evaluate all experiments
    if collector.experiments:
        files = collector.save_and_evaluate()
        print(f"\n🎉 All experiments completed!")
        print(f"📁 Files created: {files}")


if __name__ == "__main__":
    run_quick_experiment()