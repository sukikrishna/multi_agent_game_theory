"""
Manual LLM Data Collection Tool
Use this to collect LLM responses and convert them into evaluable format
"""

import json
import datetime
import os
import re
from typing import List, Dict, Any, Tuple


class ManualDataCollector:
    """Collect and process manual LLM responses"""
    
    def __init__(self):
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, game_type: str, 
                        player1_model: str, player2_model: str = "baseline"):
        """Start a new experiment session"""
        self.current_experiment = {
            'experiment_name': experiment_name,
            'game_type': game_type,
            'player1_model': player1_model,
            'player2_model': player2_model,
            'timestamp': datetime.datetime.now().isoformat(),
            'rounds': [],
            'completed': False
        }
        print(f"Started experiment: {experiment_name}")
        print(f"Game: {game_type}")
        print(f"Player 1: {player1_model}, Player 2: {player2_model}")
    
    def get_game_prompt(self, game_type: str, round_num: int, history: List[Tuple]) -> str:
        """Generate the prompt to copy-paste into LLM"""
        
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

FULL HISTORY:
""" + "\n".join([f"Round {i+1}: You: {h[0]}, Opponent: {h[1]}" 
                 for i, h in enumerate(history[-3:])])  # Show last 3 rounds
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
    
    def record_response(self, llm_response: str, opponent_action: Any = None) -> bool:
        """Record LLM response and opponent action"""
        if not self.current_experiment:
            print("No active experiment! Call start_experiment() first.")
            return False
        
        try:
            # Parse LLM response
            if self.current_experiment['game_type'] == 'colonel_blotto':
                # Parse list format [x,x,x,x,x,x]
                match = re.search(r'\[([^\]]+)\]', llm_response)
                if match:
                    numbers = [int(x.strip()) for x in match.group(1).split(',')]
                    if len(numbers) == 6 and sum(numbers) == 120:
                        player_action = numbers
                    else:
                        print(f"Invalid Blotto allocation: {numbers} (sum: {sum(numbers)})")
                        return False
                else:
                    print(f"Could not parse Blotto response: {llm_response}")
                    return False
            else:
                # Parse single number (0 or 1)
                # Extract just the number from the response
                numbers = re.findall(r'\b[01]\b', llm_response)
                if numbers:
                    player_action = int(numbers[0])  # Take first valid number
                else:
                    # Try to extract any number and see if it's 0 or 1
                    all_numbers = re.findall(r'\d+', llm_response)
                    valid_actions = [int(n) for n in all_numbers if int(n) in [0, 1]]
                    if valid_actions:
                        player_action = valid_actions[0]
                    else:
                        print(f"Could not find valid action (0 or 1) in: {llm_response}")
                        return False
            
            # Get opponent action if not provided
            if opponent_action is None:
                opponent_action = self._get_baseline_action()
            
            # Record the round
            round_data = {
                'round': len(self.current_experiment['rounds']),
                'player_action': player_action,
                'opponent_action': opponent_action,
                'llm_response': llm_response.strip()
            }
            
            self.current_experiment['rounds'].append(round_data)
            print(f"Recorded Round {round_data['round'] + 1}: You={player_action}, Opponent={opponent_action}")
            return True
            
        except Exception as e:
            print(f"Error recording response: {e}")
            return False
    
    def _get_baseline_action(self) -> Any:
        """Get baseline opponent action"""
        game_type = self.current_experiment['game_type']
        history = [(r['player_action'], r['opponent_action']) for r in self.current_experiment['rounds']]
        
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
    
    def complete_experiment(self, num_rounds: int = 10):
        """Mark experiment as complete and save"""
        if not self.current_experiment:
            print("No active experiment!")
            return
        
        if len(self.current_experiment['rounds']) < num_rounds:
            print(f"Warning: Only {len(self.current_experiment['rounds'])} rounds completed (expected {num_rounds})")
        
        self.current_experiment['completed'] = True
        self.current_experiment['total_rounds'] = len(self.current_experiment['rounds'])
        
        # Add to experiments list
        self.experiments.append(self.current_experiment.copy())
        
        print(f"Experiment completed: {self.current_experiment['experiment_name']}")
        print(f"Total rounds: {self.current_experiment['total_rounds']}")
        
        # Reset current experiment
        self.current_experiment = None
    
    def save_data(self, filename: str = None):
        """Save all experiments to file"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../../experiments/manual_llm_data/manual_experiments_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        print(f"Saved {len(self.experiments)} experiments to {filename}")
        return filename
    
    def convert_to_evaluation_format(self, filename: str = None) -> str:
        """Convert manual data to MLGym evaluation format"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../../experiments/manual_llm_data/evaluation_data_{timestamp}.json"
        
        evaluation_data = {
            'prisoners_dilemma': [],
            'battle_of_sexes': [],
            'colonel_blotto': []
        }
        
        for exp in self.experiments:
            if not exp['completed']:
                continue
                
            game_type = exp['game_type']
            
            # Convert to evaluation format
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
                    'p1_reward': 0,  # Will be calculated
                    'p2_reward': 0   # Will be calculated
                }
                session_data['results'].append(result)
            
            evaluation_data[game_type].append(session_data)
        
        # Save evaluation data
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"Converted to evaluation format: {filename}")
        return filename


def run_manual_experiment_session():
    """Interactive session for running manual experiments"""
    
    collector = ManualDataCollector()
    
    print("MANUAL LLM EXPERIMENT SESSION")
    print("=" * 50)
    
    # Get experiment details
    exp_name = input("Enter experiment name: ")
    print("\nAvailable games:")
    print("1. prisoners_dilemma")
    print("2. battle_of_sexes") 
    print("3. colonel_blotto")
    
    game_choice = input("Choose game (1-3): ")
    game_types = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
    game_type = game_types[int(game_choice) - 1]
    
    model_name = input("Enter LLM model name (e.g., 'ChatGPT-4', 'Claude-3'): ")
    num_rounds = int(input("Number of rounds to play (default 10): ") or "10")
    
    # Start experiment
    collector.start_experiment(exp_name, game_type, model_name)
    
    print(f"\nStarting {num_rounds} rounds of {game_type}")
    print("=" * 50)
    
    # Play rounds
    for round_num in range(num_rounds):
        print(f"\nROUND {round_num + 1}")
        print("-" * 30)
        
        # Get history for prompt
        history = [(r['player_action'], r['opponent_action']) for r in collector.current_experiment['rounds']]
        
        # Generate prompt
        prompt = collector.get_game_prompt(game_type, round_num, history)
        
        print("COPY THIS PROMPT TO YOUR LLM:")
        print("=" * 50)
        print(prompt)
        print("=" * 50)
        
        # Get response
        response = input("\nPaste LLM response here: ")
        
        # Record response
        success = collector.record_response(response)
        if not success:
            print("Failed to record response. Try again.")
            response = input("Re-enter response: ")
            collector.record_response(response)
    
    # Complete experiment
    collector.complete_experiment(num_rounds)
    
    # Save data
    filename = collector.save_data()
    eval_filename = collector.convert_to_evaluation_format()
    
    print(f"\nEXPERIMENT COMPLETE!")
    print(f"Raw data: {filename}")
    print(f"Evaluation data: {eval_filename}")
    
    return eval_filename


if __name__ == "__main__":
    # Run interactive session
    eval_file = run_manual_experiment_session()
    
    print(f"\nTo evaluate results, run:")
    print(f"python simple_evaluator.py {eval_file}")