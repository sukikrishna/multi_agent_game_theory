"""
Simple Evaluation Runner
Run this after collecting manual LLM data
"""

import json
import sys
import os
from game_environments import PrisonersDilemmaEnvironment, BattleOfSexesEnvironment, ColonelBlottoEnvironment


def calculate_game_rewards(game_type: str, results: list) -> list:
    """Calculate rewards for each round based on actions"""
    
    if game_type == 'prisoners_dilemma':
        payoff_matrix = {
            (0, 0): (3, 3),  # Both cooperate
            (0, 1): (0, 5),  # P1 cooperates, P2 defects
            (1, 0): (5, 0),  # P1 defects, P2 cooperates
            (1, 1): (1, 1)   # Both defect
        }
        
        for result in results:
            p1_action = result['p1_action']
            p2_action = result['p2_action']
            p1_reward, p2_reward = payoff_matrix[(p1_action, p2_action)]
            result['p1_reward'] = p1_reward
            result['p2_reward'] = p2_reward
    
    elif game_type == 'battle_of_sexes':
        payoff_matrix = {
            (0, 0): (2, 1),  # Both choose 0 (P1's preference)
            (0, 1): (0, 0),  # Miscoordination
            (1, 0): (0, 0),  # Miscoordination
            (1, 1): (1, 2)   # Both choose 1 (P2's preference)
        }
        
        for result in results:
            p1_action = result['p1_action']
            p2_action = result['p2_action']
            p1_reward, p2_reward = payoff_matrix[(p1_action, p2_action)]
            result['p1_reward'] = p1_reward
            result['p2_reward'] = p2_reward
    
    elif game_type == 'colonel_blotto':
        for result in results:
            allocation1 = result['p1_action']
            allocation2 = result['p2_action']
            
            # Count battlefield wins
            p1_wins = sum(1 for i in range(6) if allocation1[i] > allocation2[i])
            p2_wins = sum(1 for i in range(6) if allocation1[i] < allocation2[i])
            
            if p1_wins > p2_wins:
                result['p1_reward'] = 1
                result['p2_reward'] = -1
            elif p1_wins < p2_wins:
                result['p1_reward'] = -1
                result['p2_reward'] = 1
            else:
                result['p1_reward'] = 0
                result['p2_reward'] = 0
    
    return results


def evaluate_single_session(game_type: str, session_data: dict) -> dict:
    """Evaluate a single game session"""
    
    # Calculate rewards
    results = calculate_game_rewards(game_type, session_data['results'])
    
    # Create environment and calculate metrics
    if game_type == 'prisoners_dilemma':
        env = PrisonersDilemmaEnvironment()
    elif game_type == 'battle_of_sexes':
        env = BattleOfSexesEnvironment()
    elif game_type == 'colonel_blotto':
        env = ColonelBlottoEnvironment()
    else:
        raise ValueError(f"Unknown game type: {game_type}")
    
    # Populate environment history
    from game_environments import GameResult
    for result in results:
        game_result = GameResult(
            player1_action=result['p1_action'],
            player2_action=result['p2_action'],
            player1_reward=result['p1_reward'],
            player2_reward=result['p2_reward'],
            round_number=result['round']
        )
        env.history.append(game_result)
    
    # Calculate metrics
    metrics_p1 = env.calculate_metrics(player_perspective=1)
    metrics_p2 = env.calculate_metrics(player_perspective=2)
    
    return {
        'session_info': {
            'player1_strategy': session_data['player1_strategy'],
            'player2_strategy': session_data['player2_strategy'],
            'num_rounds': session_data['num_rounds'],
            'game_type': game_type
        },
        'results': results,
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


def evaluate_manual_data(data_file: str) -> dict:
    """Evaluate all manual experiment data"""
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    evaluation_results = {
        'prisoners_dilemma': [],
        'battle_of_sexes': [],
        'colonel_blotto': [],
        'summary': {}
    }
    
    print("EVALUATING MANUAL EXPERIMENT DATA")
    print("=" * 50)
    
    # Evaluate each game type
    for game_type in ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']:
        if game_type not in data or not data[game_type]:
            print(f"No data for {game_type}")
            continue
            
        print(f"\nEvaluating {game_type.replace('_', ' ').title()}...")
        game_sessions = data[game_type]
        
        for i, session in enumerate(game_sessions):
            print(f"  Session {i+1}: {session['player1_strategy']} vs {session['player2_strategy']}")
            
            try:
                result = evaluate_single_session(game_type, session)
                evaluation_results[game_type].append(result)
                
                # Print key metrics
                p1_metrics = result['metrics']['player1']
                print(f"    Player 1 Score: {p1_metrics['total_score']:.1f}")
                print(f"    Coordination Rate: {p1_metrics['coordination_rate']:.2%}")
                print(f"    Nash Deviation: {p1_metrics['nash_deviation']:.3f}")
                
            except Exception as e:
                print(f"    rror evaluating session: {e}")
    
    # Generate summary
    summary = generate_summary(evaluation_results)
    evaluation_results['summary'] = summary
    
    print("\nEVALUATION SUMMARY")
    print("=" * 30)
    for game_type, metrics in summary.items():
        if isinstance(metrics, dict) and game_type != 'overall':
            print(f"\n{game_type.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if 'rate' in metric or 'deviation' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
    
    return evaluation_results


def generate_summary(evaluation_results: dict) -> dict:
    """Generate summary statistics across all games"""
    
    summary = {}
    
    for game_type, sessions in evaluation_results.items():
        if game_type == 'summary' or not sessions:
            continue
        
        # Aggregate metrics across sessions
        total_scores = []
        coordination_rates = []
        nash_deviations = []
        regrets = []
        
        for session in sessions:
            # Get player 1 metrics (the LLM player)
            p1_metrics = session['metrics']['player1']
            total_scores.append(p1_metrics['total_score'])
            coordination_rates.append(p1_metrics['coordination_rate'])
            nash_deviations.append(p1_metrics['nash_deviation'])
            regrets.append(p1_metrics['regret'])
        
        if total_scores:  # Only if we have data
            import statistics
            summary[game_type] = {
                'num_sessions': len(sessions),
                'avg_score': statistics.mean(total_scores),
                'avg_coordination_rate': statistics.mean(coordination_rates),
                'avg_nash_deviation': statistics.mean(nash_deviations),
                'avg_regret': statistics.mean(regrets),
                'score_std': statistics.stdev(total_scores) if len(total_scores) > 1 else 0
            }
    
    # Overall summary
    if summary:
        all_scores = []
        all_coord_rates = []
        all_nash_devs = []
        
        for game_metrics in summary.values():
            all_scores.append(game_metrics['avg_score'])
            all_coord_rates.append(game_metrics['avg_coordination_rate'])
            all_nash_devs.append(game_metrics['avg_nash_deviation'])
        
        import statistics
        summary['overall'] = {
            'overall_score': statistics.mean(all_scores),
            'overall_coordination_rate': statistics.mean(all_coord_rates),
            'overall_nash_deviation': statistics.mean(all_nash_devs),
            'games_evaluated': len(summary)
        }
    
    return summary


def save_evaluation_results(results: dict, output_file: str = None):
    """Save evaluation results to file"""
    
    if output_file is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../../experiments/manual_llm_data/evaluation_results_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nEvaluation results saved to: {output_file}")
    return output_file


def print_mlgym_format_results(results: dict):
    """Print results in MLGym format for easy comparison"""
    
    summary = results.get('summary', {})
    overall = summary.get('overall', {})
    
    if overall:
        mlgym_results = {
            "Score": overall.get('overall_score', 0),
            "Coordination_Rate": overall.get('overall_coordination_rate', 0),
            "Nash_Deviation": overall.get('overall_nash_deviation', 0),
        }
        
        # Add game-specific results
        for game_type in ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']:
            if game_type in summary:
                game_summary = summary[game_type]
                prefix = game_type.upper().replace('_', '')[:3]  # PRI, BAT, COL
                mlgym_results[f"{prefix}_Score"] = game_summary.get('avg_score', 0)
                mlgym_results[f"{prefix}_Coordination"] = game_summary.get('avg_coordination_rate', 0)
        
        print("\nMLGYM FORMAT RESULTS:")
        print("=" * 30)
        print(json.dumps(mlgym_results, indent=2))
    else:
        print("\nNo overall results to display")


def main():
    """Main evaluation function"""
    
    if len(sys.argv) < 2:
        print("Usage: python simple_evaluator.py <data_file>")
        print("Example: python simple_evaluator.py ../../experiments/manual_llm_data/evaluation_data_20241215_143022.json")
        return
    
    data_file = sys.argv[1]
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    try:
        # Run evaluation
        results = evaluate_manual_data(data_file)
        
        # Save results
        output_file = save_evaluation_results(results)
        
        # Print MLGym format results
        print_mlgym_format_results(results)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()