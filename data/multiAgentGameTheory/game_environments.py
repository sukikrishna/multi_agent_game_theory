"""
Multi-Agent Game Theory Environments
Implements the three core games with evaluation metrics
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GameResult:
    """Results from a single game round"""
    player1_action: Any
    player2_action: Any
    player1_reward: float
    player2_reward: float
    round_number: int


@dataclass
class GameMetrics:
    """Comprehensive metrics for game analysis"""
    total_score: float
    coordination_rate: float
    exploitation_rate: float
    nash_deviation: float
    regret: float
    game_specific_metrics: Dict[str, float]


class GameEnvironment(ABC):
    """Base class for game theory environments"""
    
    def __init__(self, num_rounds: int = 100):
        self.num_rounds = num_rounds
        self.history: List[GameResult] = []
        self.reset()
    
    def reset(self):
        """Reset game state"""
        self.history = []
        self.current_round = 0
    
    @abstractmethod
    def get_payoff_matrix(self) -> Dict[Tuple, Tuple[float, float]]:
        """Return the payoff matrix for the game"""
        pass
    
    @abstractmethod
    def calculate_nash_prediction(self, history: List[GameResult]) -> Tuple[float, float]:
        """Calculate expected Nash equilibrium play"""
        pass
    
    def play_round(self, strategy1_func, strategy2_func) -> GameResult:
        """Play one round of the game"""
        # Get historical context for strategies
        game_history = [(r.player1_action, r.player2_action) for r in self.history]
        
        # Get actions from both players
        action1 = strategy1_func(game_history)
        action2 = strategy2_func(game_history)
        
        # Calculate rewards
        payoff_matrix = self.get_payoff_matrix()
        reward1, reward2 = payoff_matrix[(action1, action2)]
        
        # Record result
        result = GameResult(action1, action2, reward1, reward2, self.current_round)
        self.history.append(result)
        self.current_round += 1
        
        return result
    
    def play_game(self, strategy1_func, strategy2_func) -> List[GameResult]:
        """Play complete game"""
        self.reset()
        for _ in range(self.num_rounds):
            self.play_round(strategy1_func, strategy2_func)
        return self.history
    
    def calculate_metrics(self, player_perspective: int = 1) -> GameMetrics:
        """Calculate comprehensive metrics for the game"""
        if not self.history:
            return GameMetrics(0, 0, 0, 0, 0, {})
        
        # Basic scores
        if player_perspective == 1:
            scores = [r.player1_reward for r in self.history]
            actions = [r.player1_action for r in self.history]
            opponent_actions = [r.player2_action for r in self.history]
        else:
            scores = [r.player2_reward for r in self.history]
            actions = [r.player2_action for r in self.history]
            opponent_actions = [r.player1_action for r in self.history]
        
        total_score = sum(scores)
        
        # Coordination rate (mutual cooperation/beneficial outcomes)
        coordination_rate = self._calculate_coordination_rate()
        
        # Exploitation rate (times player was exploited)
        exploitation_rate = self._calculate_exploitation_rate(player_perspective)
        
        # Nash deviation
        nash_deviation = self._calculate_nash_deviation()
        
        # Regret (distance from optimal play)
        regret = self._calculate_regret(player_perspective)
        
        # Game-specific metrics
        game_specific = self._calculate_game_specific_metrics()
        
        return GameMetrics(
            total_score=total_score,
            coordination_rate=coordination_rate,
            exploitation_rate=exploitation_rate,
            nash_deviation=nash_deviation,
            regret=regret,
            game_specific_metrics=game_specific
        )
    
    @abstractmethod
    def _calculate_coordination_rate(self) -> float:
        """Calculate coordination success rate"""
        pass
    
    @abstractmethod
    def _calculate_exploitation_rate(self, player_perspective: int) -> float:
        """Calculate how often player was exploited"""
        pass
    
    @abstractmethod
    def _calculate_nash_deviation(self) -> float:
        """Calculate deviation from Nash equilibrium predictions"""
        pass
    
    @abstractmethod
    def _calculate_regret(self, player_perspective: int) -> float:
        """Calculate regret relative to optimal strategy"""
        pass
    
    @abstractmethod
    def _calculate_game_specific_metrics(self) -> Dict[str, float]:
        """Calculate game-specific metrics"""
        pass


class PrisonersDilemmaEnvironment(GameEnvironment):
    """Prisoner's Dilemma implementation"""
    
    def get_payoff_matrix(self) -> Dict[Tuple, Tuple[float, float]]:
        return {
            (0, 0): (3, 3),  # Both cooperate
            (0, 1): (0, 5),  # P1 cooperates, P2 defects
            (1, 0): (5, 0),  # P1 defects, P2 cooperates
            (1, 1): (1, 1)   # Both defect
        }
    
    def calculate_nash_prediction(self, history: List[GameResult]) -> Tuple[float, float]:
        """Nash equilibrium is always defect (1,1)"""
        return (1.0, 1.0)
    
    def _calculate_coordination_rate(self) -> float:
        """Rate of mutual cooperation"""
        cooperations = sum(1 for r in self.history if r.player1_action == 0 and r.player2_action == 0)
        return cooperations / len(self.history)
    
    def _calculate_exploitation_rate(self, player_perspective: int) -> float:
        """Rate of being exploited (cooperating while opponent defects)"""
        if player_perspective == 1:
            exploitations = sum(1 for r in self.history if r.player1_action == 0 and r.player2_action == 1)
        else:
            exploitations = sum(1 for r in self.history if r.player2_action == 0 and r.player1_action == 1)
        return exploitations / len(self.history)
    
    def _calculate_nash_deviation(self) -> float:
        """Deviation from always-defect Nash equilibrium"""
        defection_rate_p1 = sum(1 for r in self.history if r.player1_action == 1) / len(self.history)
        defection_rate_p2 = sum(1 for r in self.history if r.player2_action == 1) / len(self.history)
        return 1 - (defection_rate_p1 + defection_rate_p2) / 2
    
    def _calculate_regret(self, player_perspective: int) -> float:
        """Regret relative to best response strategy"""
        # Best response depends on opponent's strategy
        if player_perspective == 1:
            opponent_actions = [r.player2_action for r in self.history]
            player_actions = [r.player1_action for r in self.history]
        else:
            opponent_actions = [r.player1_action for r in self.history]
            player_actions = [r.player2_action for r in self.history]
        
        # Calculate regret for each round
        total_regret = 0
        payoff_matrix = self.get_payoff_matrix()
        
        for i, (player_action, opponent_action) in enumerate(zip(player_actions, opponent_actions)):
            # What player actually got
            if player_perspective == 1:
                actual_payoff = payoff_matrix[(player_action, opponent_action)][0]
                # What they could have got with best response
                best_payoff = max(payoff_matrix[(0, opponent_action)][0], 
                                payoff_matrix[(1, opponent_action)][0])
            else:
                actual_payoff = payoff_matrix[(opponent_action, player_action)][1]
                best_payoff = max(payoff_matrix[(opponent_action, 0)][1], 
                                payoff_matrix[(opponent_action, 1)][1])
            
            total_regret += best_payoff - actual_payoff
        
        return total_regret / len(self.history)
    
    def _calculate_game_specific_metrics(self) -> Dict[str, float]:
        """PD-specific metrics"""
        # Tit-for-tat success rate
        tft_violations_p1 = 0
        tft_violations_p2 = 0
        
        for i in range(1, len(self.history)):
            prev_round = self.history[i-1]
            curr_round = self.history[i]
            
            # Check if P1 played tit-for-tat
            if curr_round.player1_action != prev_round.player2_action:
                tft_violations_p1 += 1
            
            # Check if P2 played tit-for-tat
            if curr_round.player2_action != prev_round.player1_action:
                tft_violations_p2 += 1
        
        rounds_available = len(self.history) - 1
        tft_compliance_p1 = 1 - (tft_violations_p1 / rounds_available) if rounds_available > 0 else 0
        tft_compliance_p2 = 1 - (tft_violations_p2 / rounds_available) if rounds_available > 0 else 0
        
        return {
            "tit_for_tat_compliance_p1": tft_compliance_p1,
            "tit_for_tat_compliance_p2": tft_compliance_p2,
            "mutual_defection_rate": sum(1 for r in self.history if r.player1_action == 1 and r.player2_action == 1) / len(self.history)
        }


class BattleOfSexesEnvironment(GameEnvironment):
    """Battle of Sexes implementation"""
    
    def get_payoff_matrix(self) -> Dict[Tuple, Tuple[float, float]]:
        return {
            (0, 0): (2, 1),  # Both choose 0 (P1's preference)
            (0, 1): (0, 0),  # Miscoordination
            (1, 0): (0, 0),  # Miscoordination
            (1, 1): (1, 2)   # Both choose 1 (P2's preference)
        }
    
    def calculate_nash_prediction(self, history: List[GameResult]) -> Tuple[float, float]:
        """Mixed strategy Nash: P1 plays 0 with prob 2/3, P2 plays 0 with prob 1/3"""
        return (2/3, 1/3)
    
    def _calculate_coordination_rate(self) -> float:
        """Rate of successful coordination (both same choice)"""
        coordinations = sum(1 for r in self.history if r.player1_action == r.player2_action)
        return coordinations / len(self.history)
    
    def _calculate_exploitation_rate(self, player_perspective: int) -> float:
        """Rate of miscoordination (no exploitation per se in BoS)"""
        miscoordinations = sum(1 for r in self.history if r.player1_action != r.player2_action)
        return miscoordinations / len(self.history)
    
    def _calculate_nash_deviation(self) -> float:
        """Deviation from mixed strategy Nash equilibrium"""
        p1_action_0_rate = sum(1 for r in self.history if r.player1_action == 0) / len(self.history)
        p2_action_0_rate = sum(1 for r in self.history if r.player2_action == 0) / len(self.history)
        
        # Nash prediction: P1 should play 0 with prob 2/3, P2 with prob 1/3
        nash_p1_deviation = abs(p1_action_0_rate - 2/3)
        nash_p2_deviation = abs(p2_action_0_rate - 1/3)
        
        return (nash_p1_deviation + nash_p2_deviation) / 2
    
    def _calculate_regret(self, player_perspective: int) -> float:
        """Regret relative to best response strategy"""
        # Best response in BoS depends on predicting opponent coordination
        # For simplicity, calculate regret relative to always coordinating on own preference
        total_regret = 0
        payoff_matrix = self.get_payoff_matrix()
        
        for r in self.history:
            if player_perspective == 1:
                actual_payoff = r.player1_reward
                # Best would be if opponent always chose P1's preference (0)
                counterfactual_payoff = payoff_matrix[(0, 0)][0]
            else:
                actual_payoff = r.player2_reward
                # Best would be if opponent always chose P2's preference (1)
                counterfactual_payoff = payoff_matrix[(1, 1)][1]
            
            total_regret += counterfactual_payoff - actual_payoff
        
        return total_regret / len(self.history)
    
    def _calculate_game_specific_metrics(self) -> Dict[str, float]:
        """BoS-specific metrics"""
        preference_0_wins = sum(1 for r in self.history if r.player1_action == 0 and r.player2_action == 0)
        preference_1_wins = sum(1 for r in self.history if r.player1_action == 1 and r.player2_action == 1)
        
        return {
            "p1_preference_success_rate": preference_0_wins / len(self.history),
            "p2_preference_success_rate": preference_1_wins / len(self.history),
            "fairness_balance": abs(preference_0_wins - preference_1_wins) / len(self.history)
        }


class ColonelBlottoEnvironment(GameEnvironment):
    """Colonel Blotto implementation"""
    
    def __init__(self, num_rounds: int = 100, num_battlefields: int = 6, total_soldiers: int = 120):
        self.num_battlefields = num_battlefields
        self.total_soldiers = total_soldiers
        super().__init__(num_rounds)
    
    def get_payoff_matrix(self) -> Dict[Tuple, Tuple[float, float]]:
        """Blotto doesn't have a simple matrix - payoffs calculated dynamically"""
        # This is handled in calculate_blotto_payoffs
        return {}
    
    def calculate_blotto_payoffs(self, allocation1: List[int], allocation2: List[int]) -> Tuple[float, float]:
        """Calculate payoffs for Blotto game"""
        if sum(allocation1) > self.total_soldiers or sum(allocation2) > self.total_soldiers:
            # Invalid allocation
            return (-1, -1)
        
        if len(allocation1) != self.num_battlefields or len(allocation2) != self.num_battlefields:
            return (-1, -1)
        
        p1_wins = 0
        p2_wins = 0
        
        for i in range(self.num_battlefields):
            if allocation1[i] > allocation2[i]:
                p1_wins += 1
            elif allocation1[i] < allocation2[i]:
                p2_wins += 1
        
        if p1_wins > p2_wins:
            return (1, -1)
        elif p1_wins < p2_wins:
            return (-1, 1)
        else:
            return (0, 0)
    
    def play_round(self, strategy1_func, strategy2_func) -> GameResult:
        """Play one round of Blotto"""
        game_history = [(r.player1_action, r.player2_action) for r in self.history]
        
        allocation1 = strategy1_func(game_history)
        allocation2 = strategy2_func(game_history)
        
        reward1, reward2 = self.calculate_blotto_payoffs(allocation1, allocation2)
        
        result = GameResult(allocation1, allocation2, reward1, reward2, self.current_round)
        self.history.append(result)
        self.current_round += 1
        
        return result
    
    def calculate_nash_prediction(self, history: List[GameResult]) -> Tuple[float, float]:
        """Nash for Blotto is complex - use uniform random as baseline"""
        return (0.0, 0.0)  # Expected payoff in symmetric mixed strategy equilibrium
    
    def _calculate_coordination_rate(self) -> float:
        """Rate of avoiding conflict (not applicable in Blotto)"""
        return 0.0  # Blotto is purely competitive
    
    def _calculate_exploitation_rate(self, player_perspective: int) -> float:
        """Rate of losing games"""
        if player_perspective == 1:
            losses = sum(1 for r in self.history if r.player1_reward < 0)
        else:
            losses = sum(1 for r in self.history if r.player2_reward < 0)
        return losses / len(self.history)
    
    def _calculate_nash_deviation(self) -> float:
        """Simplified deviation measure - how far from uniform distribution"""
        total_deviation = 0
        uniform_allocation = self.total_soldiers / self.num_battlefields
        
        for r in self.history:
            # Calculate how far each allocation is from uniform
            p1_deviation = sum(abs(x - uniform_allocation) for x in r.player1_action) / self.total_soldiers
            p2_deviation = sum(abs(x - uniform_allocation) for x in r.player2_action) / self.total_soldiers
            total_deviation += (p1_deviation + p2_deviation) / 2
        
        return total_deviation / len(self.history)
    
    def _calculate_regret(self, player_perspective: int) -> float:
        """Regret relative to perfect information optimal play"""
        # This is complex for Blotto - simplified to win rate regret
        if player_perspective == 1:
            win_rate = sum(1 for r in self.history if r.player1_reward > 0) / len(self.history)
        else:
            win_rate = sum(1 for r in self.history if r.player2_reward > 0) / len(self.history)
        
        # Regret is distance from 50% win rate (fair game)
        return abs(0.5 - win_rate)
    
    def _calculate_game_specific_metrics(self) -> Dict[str, float]:
        """Blotto-specific metrics"""
        # Calculate allocation entropy and concentration
        total_entropy_p1 = 0
        total_entropy_p2 = 0
        total_concentration_p1 = 0
        total_concentration_p2 = 0
        
        for r in self.history:
            # Entropy (how spread out the allocation is)
            allocation1 = np.array(r.player1_action)
            allocation2 = np.array(r.player2_action)
            
            # Normalize to probabilities
            if allocation1.sum() > 0:
                prob1 = allocation1 / allocation1.sum()
                entropy1 = -np.sum(prob1 * np.log(prob1 + 1e-10))
                total_entropy_p1 += entropy1
            
            if allocation2.sum() > 0:
                prob2 = allocation2 / allocation2.sum()
                entropy2 = -np.sum(prob2 * np.log(prob2 + 1e-10))
                total_entropy_p2 += entropy2
            
            # Concentration (Gini coefficient approximation)
            concentration1 = np.sum(np.abs(allocation1 - allocation1.mean())) / (2 * allocation1.sum()) if allocation1.sum() > 0 else 0
            concentration2 = np.sum(np.abs(allocation2 - allocation2.mean())) / (2 * allocation2.sum()) if allocation2.sum() > 0 else 0
            total_concentration_p1 += concentration1
            total_concentration_p2 += concentration2
        
        return {
            "allocation_entropy_p1": total_entropy_p1 / len(self.history),
            "allocation_entropy_p2": total_entropy_p2 / len(self.history),
            "allocation_concentration_p1": total_concentration_p1 / len(self.history),
            "allocation_concentration_p2": total_concentration_p2 / len(self.history)
        }