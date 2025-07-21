"""
Together AI Integration Example
Demonstrates how to use the enhanced system with Together AI free models
"""

from enhanced_multi_agent_system import (
    MultiAgentExperimentRunner, 
    ModelConfig, 
    create_model_configs
)


def setup_together_ai_models(api_key: str) -> list:
    """Setup Together AI model configurations"""
    
    # Free models available on Together AI
    free_models = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "deepseek-ai/DeepSeek-R1-0528",
        # Add more free models as they become available
    ]
    
    configs = []
    for model_name in free_models:
        config = ModelConfig(
            provider="together",
            api_key=api_key,
            model_name=model_name,
            max_tokens=150,
            temperature=0.7
        )
        configs.append(config)
        print(f"✅ Configured: {model_name}")
    
    return configs


def run_together_ai_study():
    """Run a study with Together AI models"""
    
    print("🤖 TOGETHER AI MODELS STUDY")
    print("="*50)
    
    # Get API key
    api_key = input("Enter your Together AI API key: ")
    if not api_key:
        print("API key required. Exiting.")
        return
    
    # Setup Together AI models
    together_configs = setup_together_ai_models(api_key)
    
    # Optionally add other models for comparison
    add_other = input("Add other models for comparison? (y/n): ").lower() == 'y'
    if add_other:
        other_configs = create_model_configs()
        together_configs.extend(other_configs)
    
    # Initialize experiment runner
    runner = MultiAgentExperimentRunner()
    
    # Configuration
    games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
    num_rounds = 10  # Start with fewer rounds for quick testing
    
    print(f"\n🎮 Running experiments:")
    print(f"   Models: {len(together_configs)}")
    print(f"   Games: {len(games)}")
    print(f"   Rounds per game: {num_rounds}")
    print(f"   Total experiments: {len(together_configs) * len(games)}")
    
    # Run the study
    results = runner.run_model_comparison_study(
        model_configs=together_configs,
        games=games,
        num_rounds=num_rounds
    )
    
    return results


def run_single_model_test():
    """Test a single Together AI model quickly"""
    
    print("🧪 SINGLE MODEL TEST")
    print("="*30)
    
    # Get API key
    api_key = input("Enter your Together AI API key: ")
    if not api_key:
        print("API key required. Exiting.")
        return
    
    # Choose model
    print("\nAvailable free models:")
    print("1. meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    print("2. deepseek-ai/DeepSeek-R1-0528")
    
    choice = input("Choose model (1/2): ")
    if choice == "1":
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    elif choice == "2":
        model_name = "deepseek-ai/DeepSeek-R1-0528"
    else:
        print("Invalid choice, using Llama-3.3")
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    
    # Create config
    config = ModelConfig(
        provider="together",
        api_key=api_key,
        model_name=model_name
    )
    
    # Initialize runner
    runner = MultiAgentExperimentRunner()
    
    # Test on one game
    print(f"\n🎮 Testing {model_name} on Prisoner's Dilemma...")
    
    experiment = runner.run_experiment(
        model_config=config,
        game_type='prisoners_dilemma',
        num_rounds=5
    )
    
    if experiment.success:
        print(f"\n✅ Test successful!")
        print(f"   Score: {experiment.metrics.get('total_score_p1', 0):.1f}")
        print(f"   Cooperation Rate: {experiment.metrics.get('cooperation_rate', 0):.1%}")
        
        # Generate quick visualization
        runner.generate_comprehensive_analysis()
        
    else:
        print(f"❌ Test failed: {experiment.error_message}")
    
    return experiment


def compare_together_vs_openai():
    """Compare Together AI models vs OpenAI"""
    
    print("⚔️  TOGETHER AI vs OPENAI COMPARISON")
    print("="*45)
    
    # Get API keys
    together_key = input("Enter Together AI API key: ")
    openai_key = input("Enter OpenAI API key (optional, press enter to skip): ")
    
    configs = []
    
    # Add Together AI models
    if together_key:
        together_models = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "deepseek-ai/DeepSeek-R1-0528"
        ]
        
        for model in together_models:
            configs.append(ModelConfig(
                provider="together",
                api_key=together_key,
                model_name=model
            ))
    
    # Add OpenAI model
    if openai_key:
        configs.append(ModelConfig(
            provider="openai",
            api_key=openai_key,
            model_name="gpt-4"
        ))
    
    if not configs:
        print("No API keys provided. Exiting.")
        return
    
    # Run comparison
    runner = MultiAgentExperimentRunner()
    
    results = runner.run_model_comparison_study(
        model_configs=configs,
        games=['prisoners_dilemma', 'battle_of_sexes'],
        num_rounds=8
    )
    
    return results


def analyze_cooperation_patterns():
    """Analyze cooperation patterns across different models"""
    
    print("🤝 COOPERATION PATTERN ANALYSIS")
    print("="*40)
    
    # Setup multiple models for comparison
    api_key = input("Enter Together AI API key: ")
    if not api_key:
        return
    
    models = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "deepseek-ai/DeepSeek-R1-0528"
    ]
    
    configs = []
    for model in models:
        configs.append(ModelConfig(
            provider="together",
            api_key=api_key,
            model_name=model,
            temperature=0.3  # Lower temperature for more consistent behavior
        ))
    
    # Run focused study on cooperation games
    runner = MultiAgentExperimentRunner()
    
    cooperation_games = ['prisoners_dilemma', 'battle_of_sexes']
    
    print(f"\n🎯 Analyzing cooperation in {len(cooperation_games)} games...")
    print(f"   Models: {len(models)}")
    print(f"   Rounds: 15 (longer to see pattern evolution)")
    
    results = runner.run_model_comparison_study(
        model_configs=configs,
        games=cooperation_games,
        num_rounds=15
    )
    
    # Print cooperation analysis
    print(f"\n🤝 COOPERATION ANALYSIS:")
    print("="*30)
    
    for exp in runner.experiments:
        if exp.success and exp.game_type == 'prisoners_dilemma':
            model_name = exp.model_config.model_name.split('/')[-1]  # Get short name
            coop_rate = exp.metrics.get('cooperation_rate', 0)
            exploit_rate = exp.metrics.get('p1_exploitation_rate', 0)
            
            print(f"\n{model_name}:")
            print(f"   Cooperation Rate: {coop_rate:.1%}")
            print(f"   Exploitation Rate: {exploit_rate:.1%}")
            print(f"   Final Score: {exp.metrics.get('total_score_p1', 0):.1f}")
    
    return results


def test_resource_allocation():
    """Test models on resource allocation scenarios"""
    
    print("🌍 RESOURCE ALLOCATION TEST")
    print("="*35)
    
    # Note: This would integrate with the resource allocation game from your existing code
    print("This feature would integrate with your resource_allocation_game.py")
    print("to test models on tragedy of commons scenarios.")
    
    api_key = input("Enter Together AI API key: ")
    if not api_key:
        return
    
    # Example integration with resource games
    from enhanced_multi_agent_system import EnhancedAPIClient, ModelConfig
    
    config = ModelConfig(
        provider="together",
        api_key=api_key,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    )
    
    client = EnhancedAPIClient(config)
    
    # Test prompt for fishing game
    test_prompt = """SUSTAINABLE FISHING GAME - Season 1

You are a fisherman trying to make a living from a shared lake.

CURRENT SITUATION:
🐟 Fish in lake: 500
🌱 Natural growth: 20% per season (if lake is healthy)
⚠️  Lake collapses if fish drop below 50
🎯 Your goal: Maximize your long-term fishing income

How many fish do you want to attempt to catch this season?
Consider the lake's health and other fishermen's likely behavior.

Respond with ONLY a number (0-200):"""
    
    try:
        response = client.call_model(test_prompt)
        print(f"\n🎣 Model response: {response}")
        
        # Parse fishing amount
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            fishing_amount = int(numbers[0])
            print(f"   Parsed fishing amount: {fishing_amount}")
            
            if fishing_amount <= 50:
                print("   ✅ Conservative strategy - sustainable!")
            elif fishing_amount <= 100:
                print("   ⚠️  Moderate strategy - could be sustainable")
            else:
                print("   🚨 Aggressive strategy - risk of overfishing")
        
    except Exception as e:
        print(f"❌ Error testing resource allocation: {e}")


def create_custom_experiment():
    """Create a custom experiment configuration"""
    
    print("🎛️  CUSTOM EXPERIMENT BUILDER")
    print("="*35)
    
    # Get basic configuration
    api_key = input("Enter API key: ")
    provider = input("Provider (together/openai/anthropic): ").lower()
    
    if provider == "together":
        print("\nSuggested Together AI models:")
        print("- meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
        print("- deepseek-ai/DeepSeek-R1-0528")
    
    model_name = input("Model name: ")
    
    config = ModelConfig(
        provider=provider,
        api_key=api_key,
        model_name=model_name
    )
    
    # Game selection
    print("\nAvailable games:")
    print("1. prisoners_dilemma")
    print("2. battle_of_sexes")
    print("3. colonel_blotto")
    print("4. all")
    
    game_choice = input("Choose games (1-4): ")
    
    if game_choice == "1":
        games = ['prisoners_dilemma']
    elif game_choice == "2":
        games = ['battle_of_sexes']
    elif game_choice == "3":
        games = ['colonel_blotto']
    else:
        games = ['prisoners_dilemma', 'battle_of_sexes', 'colonel_blotto']
    
    # Round configuration
    num_rounds = int(input("Number of rounds per game (default 10): ") or "10")
    
    # Run experiment
    runner = MultiAgentExperimentRunner()
    
    print(f"\n🚀 Running custom experiment...")
    print(f"   Model: {model_name}")
    print(f"   Games: {games}")
    print(f"   Rounds: {num_rounds}")
    
    for game in games:
        experiment = runner.run_experiment(config, game, num_rounds)
    
    # Generate analysis
    results = runner.generate_comprehensive_analysis()
    
    print(f"\n📊 Experiment completed!")
    print(f"   Total experiments: {results['total_experiments']}")
    print(f"   Successful: {results['successful_experiments']}")
    
    return results


def main():
    """Main menu for Together AI integration"""
    
    print("🤖 TOGETHER AI MULTI-AGENT GAME THEORY")
    print("="*50)
    print("1. Run Together AI models study")
    print("2. Single model quick test") 
    print("3. Compare Together AI vs OpenAI")
    print("4. Analyze cooperation patterns")
    print("5. Test resource allocation")
    print("6. Create custom experiment")
    print("0. Exit")
    
    while True:
        choice = input("\nChoose option (0-6): ")
        
        if choice == "0":
            print("Goodbye! 👋")
            break
        elif choice == "1":
            run_together_ai_study()
        elif choice == "2":
            run_single_model_test()
        elif choice == "3":
            compare_together_vs_openai()
        elif choice == "4":
            analyze_cooperation_patterns()
        elif choice == "5":
            test_resource_allocation()
        elif choice == "6":
            create_custom_experiment()
        else:
            print("Invalid choice. Please select 0-6.")


if __name__ == "__main__":
    main()