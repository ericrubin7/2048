import numpy as np
import abc
import util
from game import Agent, Action
from math import inf
from random import choice


SCORE = 0
ACTION = 1

X = 0
Y = 1

# INDICES_FLATTENED = np.indices((4, 4)).flatten().reshape(-1, 2)
INDICES_FLATTENED = np.array([
    [0,0],
    [0,1],
    [0,2],
    [0,3],
    [1,0],
    [1,1],
    [1,2],
    [1,3],
    [2,0],
    [2,1],
    [2,2],
    [2,3],
    [3,0],
    [3,1],
    [3,2],
    [3,3]
])
SNAKE_INDICES = np.array([
    [3, 3],
    [3, 2],
    [3, 1],
    [3, 0],
    [2, 0],
    [2, 1],
    [2, 2],
    [2, 3],
    [1, 3],
    [1, 2],
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
])


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        score_benefit = score/10
        max_tile_benefit = max_tile*2
        score += np.count_nonzero(board == 0) * score_benefit
        score += np.count_nonzero(board == 0) * max_tile_benefit
        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
        # TODO remove
        self.init_depth = depth
        self.all_actions = set((Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))

    @abc.abstractmethod
    def get_action(self, game_state):
        return

class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        return self.minimax(game_state, self.depth, True)[ACTION]

    def minimax(self, state, depth, maximizing_player):
        if depth == 0:
            return self.evaluation_function(state), Action.STOP
        legal_moves = state.get_legal_actions(int(not maximizing_player))
        if len(legal_moves) == 0:
            return inf, Action.STOP
        if maximizing_player:
            max_value = -inf
            argmax_action = Action.STOP
            for action in legal_moves:
                child_state = state.generate_successor(action=action)
                value = self.minimax(child_state, depth, not maximizing_player)[SCORE]
                if value > max_value:
                    max_value = value
                    argmax_action = action
            if argmax_action == Action.STOP:
                argmax_action = choice(legal_moves)
            return max_value, argmax_action
        else:
            min_value = inf
            argmin_action = Action.STOP
            for action in legal_moves:
                child_state = state.generate_successor(action=action, agent_index=1)
                value = self.minimax(child_state, depth - 1, not maximizing_player)[SCORE]
                if value < min_value:
                    min_value = value
                    argmin_action = action  # actually we're not interested in it
            return min_value, argmin_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.minimax(game_state, self.depth, True, -inf, +inf)[ACTION]

    def get_action_and_update_evals(self, game_state, values):
        return self.minimax(game_state, self.depth, True, -inf, +inf, values=values)[ACTION]

    def minimax(self, state, depth, maximizing_player, alpha, beta, values=None):
        if depth == 0:
            return self.evaluation_function(state), Action.STOP
        legal_moves = state.get_legal_actions(int(not maximizing_player))
        if len(legal_moves) == 0:
            return inf, Action.STOP
        if maximizing_player:
            max_value = -inf
            argmax_action = Action.STOP
            for action in legal_moves:
                child_state = state.generate_successor(action=action)
                value = self.minimax(child_state, depth, not maximizing_player, alpha, beta)[SCORE]
                # TODO remove
                if depth == self.init_depth:
                    values[action].set(value)
                if value > max_value:
                    max_value = value
                    argmax_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            # TODO remove
            if depth == self.init_depth:
                for action in self.all_actions - set(legal_moves):
                    values[action].set(-1)
            if argmax_action == Action.STOP:
                argmax_action = choice(legal_moves)
            return max_value, argmax_action
        else:
            min_value = inf
            argmin_action = Action.STOP
            for action in legal_moves:
                child_state = state.generate_successor(action=action, agent_index=1)
                value = self.minimax(child_state, depth - 1, not maximizing_player, alpha, beta)[SCORE]
                if value < min_value:
                    min_value = value
                    argmin_action = action  # actually we're not interested in it
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value, argmin_action




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(game_state, self.depth, True)[ACTION]

    def expectimax(self, state, depth, maximizing_player):
        if depth == 0:
            return self.evaluation_function(state), Action.STOP
        legal_moves = state.get_legal_actions(int(not maximizing_player))
        if len(legal_moves) == 0:
            return inf, Action.STOP
        if maximizing_player:
            max_value = -inf
            argmax_action = Action.STOP
            for action in legal_moves:
                child_state = state.generate_successor(action=action)
                value = self.expectimax(child_state, depth, not maximizing_player)[SCORE]
                if value > max_value:
                    max_value = value
                    argmax_action = action
            if argmax_action == Action.STOP:
                argmax_action = choice(legal_moves)
            return max_value, argmax_action
        else:
            cumulative_value = 0
            for action in legal_moves:
                child_state = state.generate_successor(action=action, agent_index=1)
                cumulative_value += self.expectimax(child_state, depth - 1, not maximizing_player)[SCORE]
            return cumulative_value / len(legal_moves), choice(legal_moves)



def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # max_tile = current_game_state.max_tile
    # board = current_game_state._board
    # max_tiles_pos = np.argwhere(board == max_tile)
    score = current_game_state.score
    # side_benefit = 1    
    # x_pos, y_pos = max_tiles_pos[-1]
    # if x_pos == 0 or x_pos == 3:
    #     side_benefit *= max_tile
    # if y_pos == 0 or y_pos == 3:
    #     side_benefit *= max_tile
    # if x_pos == 3 and y_pos == 3:
    #     side_benefit *= (max_tile ** 2)
    # return score + side_benefit
    board_flattened = current_game_state._board.flatten().reshape(-1, 1)
    board_flat_w_ind = np.hstack((board_flattened, INDICES_FLATTENED))
    sorted_board_w_ind_desc = board_flat_w_ind[board_flat_w_ind[:, 0].argsort()][::-1]
    benefit = 0
    for tile, snake_ind in zip(sorted_board_w_ind_desc, SNAKE_INDICES):
        if tile[0] == 0:
            break
        if np.all(tile[1:] == snake_ind):
            benefit += tile[0]
        else:
            break
    return score+ benefit
    



# Abbreviation
better = better_evaluation_function
