"""
Quick Start Script for Game Theory Experiments
Run this to quickly set up and run your first experiment
"""

import os
import sys


def setup_directories():
    """Create necessary directories"""
    directories = [
        "../../experiments/manual_llm_data",
        "../../experiments/baseline_data",
        "../../results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def run_baseline_test():
    """Run a quick baseline test to verify everything works"""
    from game_environments import PrisonersDilemmaEnvironment, BattleOfSexesEnvironment, ColonelBlottoEnvironment
    
    print("\nRunning baseline tests...")
    
    # Test Prisoner's Dilemma
    def always_cooperate(history):
        return 0
    
    def always_defect(history):
        return 1
    
    pd_env = PrisonersDilemmaEnvironment(num_rounds=5)
    pd_results = pd_env.play_game(always_cooperate, always_defect)
    pd_metrics = pd_env.calculate_metrics(player_perspective=1)
    
    print(f"Prisoner's Dilemma test: {len(pd_results)} rounds, score: {pd_metrics.total_score}")
    
    # Test Battle of Sexes
    def prefer_0(history):
        return 0
    
    def prefer_1(history):
        return 1
    
    bos_env = BattleOfSexesEnvironment(num_rounds=5)
    bos_results = bos_env.play_game(prefer_0, prefer_1)
    bos_metrics = bos_env.calculate_metrics(player_perspective=1)
    
    print(f"Battle of Sexes test: {len(bos_results)} rounds, score: {bos_metrics.total_score}")
    
    # Test Colonel Blotto
    def uniform_strategy(history):
        return [20, 20, 20, 20, 20, 20]
    
    def concentrated_strategy(history):
        return [40, 40, 40, 0, 0, 0]
    
    blotto_env = ColonelBlottoEnvironment(num_rounds=5)
    blotto_results = blotto_env.play_game(uniform_strategy, concentrated_strategy)
    blotto_metrics = blotto_env.calculate_metrics(player_perspective=1)
    
    print(f"Colonel Blotto test: {len(blotto_results)} rounds, score: {blotto_metrics.total_score}")
    
    print("All baseline tests passed!")


def show_quick_start_guide():
    """Show quick start instructions"""
    
    print("\nGAME THEORY RESEARCH QUICK START")
    print("=" * 50)
    
    print("\nSTEP 1: Verify Setup")
    print("Your project structure should look like this:")
    print("""
multi_agent_game_theory/
├── configs/tasks/multiAgentGameTheory.yaml
├── data/multiAgentGameTheory/
│   ├── game_environments.py
│   ├── manual_data_collector.py
│   ├── simple_evaluator.py
│   └── quick_start.py (this file)
└── experiments/manual_llm_data/
    """)
    
    print("\nSTEP 2: Run Your First Experiment")
    print("To start collecting data:")
    print("  python manual_data_collector.py")
    
    print("\nSTEP 3: Evaluate Results")
    print("After collecting data:")
    print("  python simple_evaluator.py <data_file>")
    
    print("\nEXAMPLE WORKFLOW:")
    print("1. Run: python manual_data_collector.py")
    print("2. Choose: Prisoner's Dilemma (option 1)")
    print("3. Enter: ChatGPT-4 as model name")
    print("4. Set: 5 rounds for quick test")
    print("5. Copy prompts to ChatGPT and paste responses back")
    print("6. Run: python simple_evaluator.py <generated_file>")
    
    print("\nEXAMPLE PROMPT (from Prisoner's Dilemma):")
    print("""
🎮 PRISONER'S DILEMMA - Round 1

You are playing an iterated Prisoner's Dilemma game. 

RULES:
- Choose 0 (Cooperate) or 1 (Defect)
- Payoff Matrix (Your score, Opponent score):
  • Both cooperate (0,0): (3,3) - Good for both
  • You cooperate, they defect (0,1): (0,5) - You get exploited  
  • You defect, they cooperate (1,0): (5,0) - You exploit them
  • Both defect (1,1): (1,1) - Bad for both

GAME HISTORY:
This is the first round.

What is your choice for this round?
Respond with ONLY the number: 0 or 1

Your choice:
    """)
    
    print("\nWHAT YOU'LL GET:")
    print("- Coordination rates across games")
    print("- Nash equilibrium deviations")
    print("- Exploitation and regret metrics")
    print("- MLGym-format results for research")
    
    print("\nSCALING UP:")
    print("- Test all 3 games (PD, BoS, Blotto)")
    print("- Try different LLM models")
    print("- Experiment with different prompts")
    print("- Run longer experiments (10-20 rounds)")
    
    print(f"\nReady to start! Run: python manual_data_collector.py")


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    try:
        import numpy as np
        print("numpy available")
    except ImportError:
        print("numpy not found - install with: pip install numpy")
        return False
    
    try:
        import json
        print("json available")
    except ImportError:
        print("json not found")
        return False
    
    try:
        from game_environments import PrisonersDilemmaEnvironment
        print("game_environments module available")
    except ImportError:
        print("game_environments module not found - check file location")
        return False
    
    print("All dependencies available!")
    return True


def main():
    """Main function"""
    
    print("🎮 MULTI-AGENT GAME THEORY RESEARCH SETUP")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease fix dependency issues before continuing")
        return
    
    # Setup directories
    setup_directories()
    
    # Run baseline tests
    try:
        run_baseline_test()
    except Exception as e:
        print(f"Baseline test failed: {e}")
        print("Check that all code files are in the correct locations")
        return
    
    # Show quick start guide
    show_quick_start_guide()


if __name__ == "__main__":
    main()