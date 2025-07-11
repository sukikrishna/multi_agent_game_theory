"""
Multi-Agent Game Theory Evaluation Script
Comprehensive evaluation across all three games with detailed metrics
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from game_environments import (
    PrisonersDilemmaEnvironment, 
    BattleOfSexesEnvironment, 
    ColonelBlottoEnvironment
)
from strategies import MultiAgentStrategies, GameDataCollector
import itertools


def create_baseline_strategies():
    """Create baseline strategies for comparison"""
    
    def random_pd_strategy(history):
        return np.random.choice([0, 1])
    
    def tit_for_tat_strategy(history):
        if not history:
            return 0
        return history[-1][1]  # Copy opponent's last move
    
    def always_cooperate_strategy(history):
        return 0
    
    def always_defect_strategy(history):
        return 1
    
    def random_bos_strategy(history):
        return np.random.choice([0, 1])
    
    def stubborn_bos_strategy(history):
        return 0  # Always choose own preference
    
    def alternating_bos_strategy(history):
        return len(history) % 2
    
    def random_blotto_strategy(history):
        allocation = np.random.multinomial(120, [1/6]*6)
        return allocation.tolist()
    
    def uniform_blotto_strategy(history):
        return [20] * 6
    
    def concentrated_blotto_strategy(history):
        # Focus on 3 battlefields
        return [40, 40, 40, 0, 0, 0]
    
    return {
        'prisoners_dilemma': {
            'random': random_pd_strategy,
            'tit_for_tat': tit_for_tat_strategy,
            'always_cooperate': always_cooperate_strategy,
            'always_defect': always_defect_strategy
        },
        'battle_of_sexes': {
            'random': random_bos_strategy,
            'stubborn': stubborn_bos_strategy,
            'alternating': alternating_bos_strategy
        },
        'colonel_blotto': {
            'random': random_blotto_strategy,
            'uniform': uniform_blotto_strategy,
            'concentrated': concentrated_blotto_strategy
        }
    }


def run_single_game_evaluation(game_env, strategy1, strategy2, strategy1_name, strategy2_name, num_rounds=100):
    """Run a single game evaluation between two strategies"""
    
    # Run the game
    results = game_env.play_game(strategy1, strategy2)
    
    # Calculate metrics for both players
    metrics_p1 = game_env.calculate_metrics(player_perspective=1)
    metrics_p2 = game_env.calculate_metrics(player_perspective=2)
    
    # Compile detailed results
    game_data = {
        'player1_strategy': strategy1_name,
        'player2_strategy': strategy2_name,
        'num_rounds': num_rounds,
        'game_results': [
            {
                'round': r.round_number,
                'p1_action': r.player1_action,
                'p2_action': r.player2_action,
                'p1_reward': r.player1_reward,
                'p2_reward': r.player2_reward
            } for r in results
        ],
        'metrics': {
            'player1': {
                'total_score': metrics_p1.total_score,
                'coordination_rate': metrics_p1.coordination_rate,
                'exploitation_rate': metrics_p1.exploitation_rate,
                'nash_deviation': metrics_p1.nash_deviation,
                'regret': metrics_p1.regret,
                'game_specific': metrics_p1.game_specific_metrics
            },
            'player2': {
                'total_score': metrics_p2.total_score,
                'coordination_rate': metrics_p2.coordination_rate,
                'exploitation_rate': metrics_p2.exploitation_rate,
                'nash_deviation': metrics_p2.nash_deviation,
                'regret': metrics_p2.regret,
                'game_specific': metrics_p2.game_specific_metrics
            }
        }
    }
    
    return game_data


def run_comprehensive_evaluation(user_strategies=None, num_rounds=100, num_repetitions=5):
    """
    Run comprehensive evaluation across all games and strategy combinations
    
    Args:
        user_strategies: Dictionary of user-defined strategies for each game
        num_rounds: Number of rounds per game
        num_repetitions: Number of times to repeat each matchup
    """
    
    # Get baseline strategies
    baseline_strategies = create_baseline_strategies()
    
    # Initialize user strategies if not provided
    if user_strategies is None:
        user_agent = MultiAgentStrategies()
        user_strategies = {
            'prisoners_dilemma': {
                'user_strategy': user_agent.prisoners_dilemma_strategy
            },
            'battle_of_sexes': {
                'user_strategy': user_agent.battle_of_sexes_strategy
            },
            'colonel_blotto': {
                'user_strategy': user_agent.colonel_blotto_strategy
            }
        }
    
    # Combine baseline and user strategies
    all_strategies = {}
    for game_type in baseline_strategies:
        all_strategies[game_type] = {**baseline_strategies[game_type], **user_strategies.get(game_type, {})}
    
    # Initialize environments
    environments = {
        'prisoners_dilemma': PrisonersDilemmaEnvironment(num_rounds),
        'battle_of_sexes': BattleOfSexesEnvironment(num_rounds),
        'colonel_blotto': ColonelBlottoEnvironment(num_rounds)
    }
    
    # Data collector
    collector = GameDataCollector()
    
    # Results storage
    evaluation_results = {
        'prisoners_dilemma': [],
        'battle_of_sexes': [],
        'colonel_blotto': []
    }
    
    print("🎮 Starting comprehensive game theory evaluation...")
    
    for game_type, env in environments.items():
        print(f"\n📊 Evaluating {game_type.replace('_', ' ').title()}...")
        
        strategies = all_strategies[game_type]
        strategy_names = list(strategies.keys())
        
        # Test all strategy combinations
        for i, strategy1_name in enumerate(strategy_names):
            for j, strategy2_name in enumerate(strategy_names):
                if i <= j:  # Avoid duplicate symmetric matchups
                    continue
                
                print(f"  {strategy1_name} vs {strategy2_name}")
                
                # Run multiple repetitions
                repetition_results = []
                for rep in range(num_repetitions):
                    game_data = run_single_game_evaluation(
                        env, 
                        strategies[strategy1_name], 
                        strategies[strategy2_name],
                        strategy1_name,
                        strategy2_name,
                        num_rounds
                    )
                    repetition_results.append(game_data)
                
                # Aggregate repetition results
                aggregated_data = aggregate_repetition_results(repetition_results)
                evaluation_results[game_type].append(aggregated_data)
                
                # Collect for analysis
                collector.collect_game_session(
                    game_type,
                    aggregated_data['game_results'],
                    aggregated_data['metrics'],
                    strategy1_name,
                    strategy2_name
                )
    
    # Generate comprehensive metrics
    final_metrics = generate_final_metrics(evaluation_results)
    
    # Export data
    collector.export_for_analysis('comprehensive_game_data.json')
    
    return {
        'detailed_results': evaluation_results,
        'final_metrics': final_metrics,
        'summary': collector.get_summary_stats()
    }


def aggregate_repetition_results(repetition_results: List[Dict]) -> Dict:
    """Aggregate results across multiple repetitions"""
    
    # Average metrics across repetitions
    p1_metrics_avg = {}
    p2_metrics_avg = {}
    
    metric_keys = ['total_score', 'coordination_rate', 'exploitation_rate', 'nash_deviation', 'regret']
    
    for key in metric_keys:
        p1_values = [r['metrics']['player1'][key] for r in repetition_results]
        p2_values = [r['metrics']['player2'][key] for r in repetition_results]
        
        p1_metrics_avg[key] = {
            'mean': np.mean(p1_values),
            'std': np.std(p1_values),
            'min': np.min(p1_values),
            'max': np.max(p1_values)
        }
        
        p2_metrics_avg[key] = {
            'mean': np.mean(p2_values),
            'std': np.std(p2_values),
            'min': np.min(p2_values),
            'max': np.max(p2_values)
        }
    
    # Aggregate game-specific metrics
    p1_game_specific = {}
    p2_game_specific = {}
    
    if repetition_results and 'game_specific' in repetition_results[0]['metrics']['player1']:
        game_specific_keys = repetition_results[0]['metrics']['player1']['game_specific'].keys()
        
        for key in game_specific_keys:
            p1_values = [r['metrics']['player1']['game_specific'][key] for r in repetition_results]
            p2_values = [r['metrics']['player2']['game_specific'][key] for r in repetition_results]
            
            p1_game_specific[key] = {
                'mean': np.mean(p1_values),
                'std': np.std(p1_values)
            }
            
            p2_game_specific[key] = {
                'mean': np.mean(p2_values),
                'std': np.std(p2_values)
            }
    
    return {
        'player1_strategy': repetition_results[0]['player1_strategy'],
        'player2_strategy': repetition_results[0]['player2_strategy'],
        'num_repetitions': len(repetition_results),
        'num_rounds': repetition_results[0]['num_rounds'],
        'game_results': repetition_results[0]['game_results'],  # Use first repetition for detailed results
        'metrics': {
            'player1': {**p1_metrics_avg, 'game_specific': p1_game_specific},
            'player2': {**p2_metrics_avg, 'game_specific': p2_game_specific}
        }
    }


def generate_final_metrics(evaluation_results: Dict) -> Dict:
    """Generate final aggregated metrics across all games"""
    
    final_metrics = {}
    
    for game_type, results in evaluation_results.items():
        if not results:
            continue
        
        # Find user strategy performance
        user_results = [r for r in results if 'user_strategy' in [r['player1_strategy'], r['player2_strategy']]]
        
        if user_results:
            # Calculate average performance when user is player 1 or player 2
            user_as_p1 = [r for r in user_results if r['player1_strategy'] == 'user_strategy']
            user_as_p2 = [r for r in user_results if r['player2_strategy'] == 'user_strategy']
            
            user_scores = []
            user_coordination_rates = []
            user_nash_deviations = []
            user_regrets = []
            
            for r in user_as_p1:
                user_scores.append(r['metrics']['player1']['total_score']['mean'])
                user_coordination_rates.append(r['metrics']['player1']['coordination_rate']['mean'])
                user_nash_deviations.append(r['metrics']['player1']['nash_deviation']['mean'])
                user_regrets.append(r['metrics']['player1']['regret']['mean'])
            
            for r in user_as_p2:
                user_scores.append(r['metrics']['player2']['total_score']['mean'])
                user_coordination_rates.append(r['metrics']['player2']['coordination_rate']['mean'])
                user_nash_deviations.append(r['metrics']['player2']['nash_deviation']['mean'])
                user_regrets.append(r['metrics']['player2']['regret']['mean'])
            
            final_metrics[game_type] = {
                'average_score': np.mean(user_scores) if user_scores else 0,
                'average_coordination_rate': np.mean(user_coordination_rates) if user_coordination_rates else 0,
                'average_nash_deviation': np.mean(user_nash_deviations) if user_nash_deviations else 0,
                'average_regret': np.mean(user_regrets) if user_regrets else 0,
                'num_matchups': len(user_results)
            }
        else:
            final_metrics[game_type] = {
                'average_score': 0,
                'average_coordination_rate': 0,
                'average_nash_deviation': 0,
                'average_regret': 0,
                'num_matchups': 0
            }
    
    # Calculate overall performance metrics
    overall_score = np.mean([metrics['average_score'] for metrics in final_metrics.values()])
    overall_coordination = np.mean([metrics['average_coordination_rate'] for metrics in final_metrics.values()])
    overall_nash_deviation = np.mean([metrics['average_nash_deviation'] for metrics in final_metrics.values()])
    overall_regret = np.mean([metrics['average_regret'] for metrics in final_metrics.values()])
    
    final_metrics['overall'] = {
        'Score': overall_score,
        'Coordination_Rate': overall_coordination,
        'Nash_Deviation': overall_nash_deviation,
        'Regret': overall_regret
    }
    
    return final_metrics


def main_evaluation():
    """Main evaluation function - matches MLGym evaluation pattern"""
    
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(
        user_strategies=None,  # Uses default strategies from MultiAgentStrategies
        num_rounds=100,
        num_repetitions=3
    )
    
    # Print results in MLGym format
    metrics = results['final_metrics']['overall']
    print(json.dumps(metrics))
    
    # Also save detailed results
    with open('detailed_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return metrics


if __name__ == "__main__":
    main_evaluation()