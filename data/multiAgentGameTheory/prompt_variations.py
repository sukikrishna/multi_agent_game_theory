"""
Alternative Prompt Variations for Game Theory Experiments
Test different framings to see how they affect behavior
"""

class PromptVariations:
    """Collection of different prompt styles for each game"""
    
    @staticmethod
    def get_prompt_variants(game_type: str) -> dict:
        """Get all prompt variants for a game type"""
        
        if game_type == "prisoners_dilemma":
            return PromptVariations.get_pd_variants()
        elif game_type == "battle_of_sexes":
            return PromptVariations.get_bos_variants()
        elif game_type == "colonel_blotto":
            return PromptVariations.get_blotto_variants()
        else:
            raise ValueError(f"Unknown game type: {game_type}")
    
    @staticmethod
    def get_pd_variants() -> dict:
        """Prisoner's Dilemma prompt variations"""
        
        return {
            "standard": lambda round_num, history: f"""PRISONER'S DILEMMA - Round {round_num + 1}

You are playing an iterated Prisoner's Dilemma game. 

RULES:
- Choose 0 (Cooperate) or 1 (Defect)
- Payoff Matrix (Your score, Opponent score):
  • Both cooperate (0,0): (3,3) - Good for both
  • You cooperate, they defect (0,1): (0,5) - You get exploited  
  • You defect, they cooperate (1,0): (5,0) - You exploit them
  • Both defect (1,1): (1,1) - Bad for both

GAME HISTORY:
{PromptVariations._format_history(history) if history else "This is the first round."}

What is your choice for this round?
Respond with ONLY the number: 0 or 1

Your choice:""",

            "competitive": lambda round_num, history: f"""STRATEGIC COMPETITION - Round {round_num + 1}

You are in a competitive strategic game. Your goal is to maximize YOUR score.

SCORING SYSTEM:
- Choose 0 (Cooperate) or 1 (Defect)
- Points you earn:
  • If you cooperate and opponent cooperates: +3 points
  • If you cooperate and opponent defects: +0 points (YOU GET EXPLOITED!)
  • If you defect and opponent cooperates: +5 points (YOU WIN!)
  • If you defect and opponent defects: +1 point

GAME HISTORY:
{PromptVariations._format_history(history) if history else "This is the first round."}

Choose your strategy to maximize YOUR points!
Respond with ONLY: 0 or 1

Your choice:""",

            "cooperative": lambda round_num, history: f"""MUTUAL COOPERATION GAME - Round {round_num + 1}

You are working with a partner to achieve the best outcome for both of you.

COLLABORATION RULES:
- Choose 0 (Work Together) or 1 (Work Alone)
- Outcomes:
  • Both work together (0,0): (3,3) - Great success for both!
  • One works together, other works alone: (0,5) or (5,0) - Unfair outcome
  • Both work alone (1,1): (1,1) - Poor outcome for both

PARTNERSHIP HISTORY:
{PromptVariations._format_history(history, labels=["Work Together", "Work Alone"]) if history else "This is your first collaboration."}

How do you want to approach this round?
Respond with ONLY: 0 or 1

Your choice:""",

            "trust": lambda round_num, history: f"""TRUST BUILDING EXERCISE - Round {round_num + 1}

You are building a relationship with another person. Trust is key to success.

TRUST DYNAMICS:
- Choose 0 (Trust) or 1 (Distrust)
- Relationship outcomes:
  • Mutual trust (0,0): (3,3) - Strong relationship
  • You trust, they distrust (0,1): (0,5) - You get hurt
  • You distrust, they trust (1,0): (5,0) - You take advantage
  • Mutual distrust (1,1): (1,1) - Damaged relationship

RELATIONSHIP HISTORY:
{PromptVariations._format_history(history, labels=["Trust", "Distrust"]) if history else "This is your first interaction."}

What is your approach this round?
Respond with ONLY: 0 or 1

Your choice:""",

            "evolutionary": lambda round_num, history: f"""EVOLUTIONARY SURVIVAL GAME - Round {round_num + 1}

You are a species trying to survive in an ecosystem with another species.

SURVIVAL STRATEGIES:
- Choose 0 (Share Resources) or 1 (Compete for Resources)
- Survival outcomes:
  • Both share (0,0): (3,3) - Ecosystem thrives
  • You share, they compete (0,1): (0,5) - You struggle to survive
  • You compete, they share (1,0): (5,0) - You dominate
  • Both compete (1,1): (1,1) - Ecosystem degrades

ECOSYSTEM HISTORY:
{PromptVariations._format_history(history, labels=["Share", "Compete"]) if history else "The ecosystem is just forming."}

What survival strategy do you choose?
Respond with ONLY: 0 or 1

Your choice:"""
        }
    
    @staticmethod
    def get_bos_variants() -> dict:
        """Battle of Sexes prompt variations"""
        
        return {
            "standard": lambda round_num, history: f"""BATTLE OF SEXES - Round {round_num + 1}

You are playing a coordination game with conflicting preferences.

RULES:
- Choose 0 (Your preference) or 1 (Opponent's preference)
- Payoff Matrix (Your score, Opponent score):
  • Both choose 0: (2,1) - You get your preference
  • Both choose 1: (1,2) - Opponent gets their preference
  • Different choices: (0,0) - Nobody gets anything

GOAL: Coordinate on the same choice, preferably yours!

GAME HISTORY:
{PromptVariations._format_history(history) if history else "This is the first round."}

What is your choice for this round?
Respond with ONLY: 0 or 1

Your choice:""",

            "restaurant": lambda round_num, history: f"""DINNER PLANS - Round {round_num + 1}

You and your partner are deciding where to go for dinner tonight.

RESTAURANT CHOICES:
- Choose 0 (Italian Restaurant - your favorite) or 1 (Sushi Restaurant - their favorite)
- Satisfaction levels:
  • Both choose Italian: You=Very Happy (2), Partner=Happy (1)
  • Both choose Sushi: You=Happy (1), Partner=Very Happy (2)
  • Different choices: You both stay home hungry (0,0)

DINING HISTORY:
{PromptVariations._format_history(history, labels=["Italian", "Sushi"]) if history else "This is your first dinner plan together."}

Where do you want to go for dinner?
Respond with ONLY: 0 or 1

Your choice:""",

            "meeting": lambda round_num, history: f"""MEETING COORDINATION - Round {round_num + 1}

You need to coordinate a meeting time with a colleague.

TIME PREFERENCES:
- Choose 0 (Morning - your preferred time) or 1 (Afternoon - their preferred time)
- Productivity outcomes:
  • Both choose Morning: Your productivity=High (2), Their productivity=Low (1)
  • Both choose Afternoon: Your productivity=Low (1), Their productivity=High (2)
  • Different choices: No meeting happens, no work gets done (0,0)

SCHEDULING HISTORY:
{PromptVariations._format_history(history, labels=["Morning", "Afternoon"]) if history else "This is your first scheduling attempt."}

What time do you suggest?
Respond with ONLY: 0 or 1

Your choice:""",

            "leadership": lambda round_num, history: f"""LEADERSHIP COORDINATION - Round {round_num + 1}

You and a colleague need to decide who leads this project.

LEADERSHIP ROLES:
- Choose 0 (You lead) or 1 (They lead)
- Performance outcomes:
  • You lead, they follow: Your satisfaction=High (2), Their satisfaction=Okay (1)
  • They lead, you follow: Your satisfaction=Okay (1), Their satisfaction=High (2)
  • No coordination: Project fails, both unsatisfied (0,0)

COLLABORATION HISTORY:
{PromptVariations._format_history(history, labels=["You Lead", "They Lead"]) if history else "This is your first project together."}

Who should lead this project?
Respond with ONLY: 0 or 1

Your choice:"""
        }
    
    @staticmethod
    def get_blotto_variants() -> dict:
        """Colonel Blotto prompt variations"""
        
        return {
            "standard": lambda round_num, history: f"""COLONEL BLOTTO - Round {round_num + 1}

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

{PromptVariations._format_blotto_history(history) if history else "This is the first battle."}

Allocate your 120 soldiers across 6 battlefields.
Respond with ONLY 6 numbers in brackets that sum to 120.
Example: [25,25,25,25,10,10]

Your allocation:""",

            "business": lambda round_num, history: f"""MARKET COMPETITION - Round {round_num + 1}

You are a CEO allocating marketing budget across 6 regional markets.

BUSINESS SCENARIO:
- You have $120M marketing budget to allocate across 6 regions
- You win a region if you spend MORE than your competitor there
- Win the quarter if you dominate MORE regions than your competitor
- Quarterly performance: +1 for winning, -1 for losing, 0 for tie

REGIONS: North, South, East, West, International, Online

{PromptVariations._format_blotto_history(history, labels=["Budget", "Competitor Budget"]) if history else "This is your first quarterly allocation."}

How do you allocate your $120M budget? (in millions)
Respond with ONLY 6 numbers in brackets that sum to 120.
Example: [25,25,25,25,10,10]

Your allocation:""",

            "political": lambda round_num, history: f"""CAMPAIGN STRATEGY - Round {round_num + 1}

You are a campaign manager allocating resources across 6 key districts.

CAMPAIGN SCENARIO:
- You have 120 campaign workers to deploy across 6 districts
- You win a district if you deploy MORE workers than your opponent
- Win the election if you win MORE districts than your opponent
- Election outcome: +1 for victory, -1 for defeat, 0 for tie

DISTRICTS: Urban, Suburban, Rural, Youth, Senior, Swing

{PromptVariations._format_blotto_history(history, labels=["Your Workers", "Opponent Workers"]) if history else "This is your first deployment decision."}

How do you deploy your 120 campaign workers?
Respond with ONLY 6 numbers in brackets that sum to 120.
Example: [25,25,25,25,10,10]

Your deployment:""",

            "sports": lambda round_num, history: f"""SPORTS STRATEGY - Round {round_num + 1}

You are a coach allocating 120 minutes of practice time across 6 skill areas.

TRAINING SCENARIO:
- You have 120 minutes of practice time to allocate
- You outperform opponents in areas where you practice MORE
- Win the game if you dominate MORE skill areas than your opponent
- Game result: +1 for win, -1 for loss, 0 for tie

SKILL AREAS: Offense, Defense, Conditioning, Strategy, Teamwork, Fundamentals

{PromptVariations._format_blotto_history(history, labels=["Your Practice", "Opponent Practice"]) if history else "This is your first practice session."}

How do you allocate your 120 practice minutes?
Respond with ONLY 6 numbers in brackets that sum to 120.
Example: [25,25,25,25,10,10]

Your allocation:""",

            "research": lambda round_num, history: f"""RESEARCH FUNDING - Round {round_num + 1}

You are a research director allocating $120M across 6 research areas.

FUNDING SCENARIO:
- You have $120M research budget to distribute
- You lead in areas where you invest MORE than competitors
- Win the innovation race if you lead in MORE areas than competitors
- Innovation outcome: +1 for leadership, -1 for falling behind, 0 for tie

RESEARCH AREAS: AI, Biotech, Energy, Materials, Computing, Robotics

{PromptVariations._format_blotto_history(history, labels=["Your Funding", "Competitor Funding"]) if history else "This is your first funding cycle."}

How do you allocate your $120M research budget? (in millions)
Respond with ONLY 6 numbers in brackets that sum to 120.
Example: [25,25,25,25,10,10]

Your allocation:"""
        }
    
    @staticmethod
    def _format_history(history: list, labels: list = None) -> str:
        """Format game history for display"""
        if labels is None:
            labels = ["0", "1"]
        
        return "\n".join([
            f"Round {i+1}: You chose {labels[h[0]]}, Opponent chose {labels[h[1]]}" 
            for i, h in enumerate(history)
        ])
    
    @staticmethod
    def _format_blotto_history(history: list, labels: list = None) -> str:
        """Format Blotto history for display"""
        if labels is None:
            labels = ["Your allocation", "Opponent allocation"]
        
        if not history:
            return ""
        
        # Show last few rounds
        recent_history = history[-3:] if len(history) > 3 else history
        
        history_str = f"\nRECENT HISTORY:\n"
        for i, h in enumerate(recent_history):
            round_num = len(history) - len(recent_history) + i + 1
            history_str += f"Round {round_num}: {labels[0]}: {h[0]}, {labels[1]}: {h[1]}\n"
        
        return history_str


# Modified AutomatedDataCollector that supports prompt variations
class VariantDataCollector:
    """Extended collector that supports different prompt variants"""
    
    def __init__(self, api_config, prompt_variant: str = "standard"):
        from automated_collector import AutomatedDataCollector
        self.base_collector = AutomatedDataCollector(api_config)
        self.prompt_variant = prompt_variant
        self.prompt_variants = {}
    
    def start_experiment(self, experiment_name: str, game_type: str, 
                        player1_model: str, num_rounds: int = 10):
        """Start experiment with specific prompt variant"""
        # Load prompt variants for this game
        self.prompt_variants = PromptVariations.get_prompt_variants(game_type)
        
        # Start base experiment
        self.base_collector.start_experiment(
            f"{experiment_name}_{self.prompt_variant}",
            game_type,
            f"{player1_model}_{self.prompt_variant}",
            num_rounds
        )
        
        print(f"🎭 Using prompt variant: {self.prompt_variant}")
    
    def get_game_prompt(self, game_type: str, round_num: int, history: list) -> str:
        """Get prompt using the specified variant"""
        if self.prompt_variant in self.prompt_variants:
            prompt_func = self.prompt_variants[self.prompt_variant]
            return prompt_func(round_num, history)
        else:
            # Fallback to standard
            return self.base_collector.get_game_prompt(game_type, round_num, history)
    
    def run_experiment(self) -> bool:
        """Run experiment with variant prompt"""
        # Override the prompt generation method
        original_method = self.base_collector.get_game_prompt
        self.base_collector.get_game_prompt = self.get_game_prompt
        
        # Run the experiment
        result = self.base_collector.run_experiment()
        
        # Restore original method
        self.base_collector.get_game_prompt = original_method
        
        return result
    
    def save_and_evaluate(self, output_dir: str = "../../experiments/variant_data") -> dict:
        """Save variant experiment data"""
        return self.base_collector.save_and_evaluate(output_dir)


def run_prompt_comparison_study():
    """Run a comprehensive study comparing different prompt variants"""
    
    # Set up API configuration
    api_key = input("Enter your API key: ")
    provider = input("Enter provider (openai/together): ").lower()
    
    if provider == "openai":
        from automated_collector import get_openai_config
        config = get_openai_config(api_key, "gpt-4")
    elif provider == "together":
        from automated_collector import get_together_config
        model = input("Enter model name (or press enter for Llama-2-70b): ") or "meta-llama/Llama-2-70b-chat-hf"
        config = get_together_config(api_key, model)
    else:
        print("❌ Unsupported provider")
        return
    
    # Games and their available variants
    games_variants = {
        'prisoners_dilemma': ['standard', 'competitive', 'cooperative', 'trust', 'evolutionary'],
        'battle_of_sexes': ['standard', 'restaurant', 'meeting', 'leadership'],
        'colonel_blotto': ['standard', 'business', 'political', 'sports', 'research']
    }
    
    all_results = {}
    
    print(f"\n🎭 PROMPT VARIATION STUDY")
    print(f"Model: {config.model_name}")
    print(f"Total experiments: {sum(len(variants) for variants in games_variants.values())}")
    
    # Run experiments for each game and variant
    for game_type, variants in games_variants.items():
        game_results = {}
        
        print(f"\n🎮 Testing {game_type.replace('_', ' ').title()}...")
        
        for variant in variants:
            print(f"\n🎭 Variant: {variant}")
            
            # Create collector with specific variant
            collector = VariantDataCollector(config, variant)
            
            # Run experiment
            collector.start_experiment(
                experiment_name=f"prompt_study_{game_type}",
                game_type=game_type,
                player1_model=config.model_name,
                num_rounds=10
            )
            
            success = collector.run_experiment()
            
            if success:
                files = collector.save_and_evaluate()
                game_results[variant] = files
                print(f"✅ {variant} completed")
            else:
                print(f"❌ {variant} failed")
        
        all_results[game_type] = game_results
    
    # Generate comparison report
    generate_variant_comparison_report(all_results)
    
    print(f"\n🎉 Prompt variation study completed!")
    return all_results


def generate_variant_comparison_report(results: dict):
    """Generate a report comparing different prompt variants"""
    
    print(f"\n" + "="*80)
    print("🎭 PROMPT VARIATION COMPARISON REPORT")
    print("="*80)
    
    for game_type, variants in results.items():
        print(f"\n🎮 {game_type.replace('_', ' ').title().upper()}")
        print("-" * 50)
        
        for variant, files in variants.items():
            if 'results' in files:
                try:
                    # Load results file
                    with open(files['results'], 'r') as f:
                        import json
                        result_data = json.load(f)
                    
                    # Extract key metrics
                    summary = result_data.get('summary', {})
                    if game_type in summary:
                        metrics = summary[game_type]
                        print(f"\n{variant.upper()}:")
                        print(f"  Score: {metrics.get('avg_score', 0):.1f}")
                        print(f"  Coordination: {metrics.get('avg_coordination_rate', 0):.1%}")
                        print(f"  Nash Deviation: {metrics.get('avg_nash_deviation', 0):.3f}")
                        print(f"  Regret: {metrics.get('avg_regret', 0):.2f}")
                
                except Exception as e:
                    print(f"\n{variant.upper()}: Error loading results - {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("🎭 PROMPT VARIATION TESTING")
    print("1. Run single variant experiment")
    print("2. Run comprehensive comparison study")
    
    choice = input("Choose option (1/2): ")
    
    if choice == "1":
        # Single variant test
        game = input("Game type (prisoners_dilemma/battle_of_sexes/colonel_blotto): ")
        variants = PromptVariations.get_prompt_variants(game)
        print(f"Available variants: {list(variants.keys())}")
        variant = input("Choose variant: ")
        
        # Run single experiment with chosen variant
        api_key = input("Enter API key: ")
        from automated_collector import get_openai_config
        config = get_openai_config(api_key)
        
        collector = VariantDataCollector(config, variant)
        collector.start_experiment(f"test_{variant}", game, config.model_name, 5)
        collector.run_experiment()
        collector.save_and_evaluate()
        
    elif choice == "2":
        # Full comparison study
        run_prompt_comparison_study()
    
    else:
        print("Invalid choice")