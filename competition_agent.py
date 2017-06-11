"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))     
    
    if len(opponent_moves) == 1 and opponent_moves[0] in my_moves:
        return float("inf")        
    
    points = 0.
    borders = [0, game.width-1, game.height-1]
    x1, y1 = game.get_player_location(player)    
    
    if x1 in borders:        
        points -= 5
    if y1 in borders:        
        points -= 5
        
    w, h = game.width / 2., game.height / 2.
    points -= float( ((h - y1)**2 + (w - x1)**2)**0.5  )
    
    moves = len(game.get_blank_spaces())
    if (moves > 10):        
        x2, y2 = game.get_player_location(game.get_opponent(player))
        distance = (x1+y1)**2 + (x2+y2)**2   
        if x2 in borders or y2 in borders:
            points += 5
            return points + float(len(my_moves) - len(opponent_moves)*3 + 3/(distance/game.width))
        else:
            return points + float(len(my_moves) - len(opponent_moves)*2 - (distance/game.width))                    
    else:        
        return float(len(my_moves)*3 - len(opponent_moves))


class CustomPlayer:
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
    """

    def __init__(self, data=None, timeout=1.):
        self.score = custom_score
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[random.randint(0, len(legal_moves)-1)]            
            
        try:
            depth = 1
            while True:
                move = self.alphabeta(game, depth)                
                if move != (-1, -1):
                    best_move = move
                depth+=1

        except SearchTimeout:
            pass
            
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()                              

        if depth == 0:
            return (-1, -1)
        
        _, best_move = self.alphabeta_max(game, depth, alpha, beta)
        return best_move

    
    def alphabeta_max(self, game, depth, alpha, beta): 
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        best_move = (-1, -1)
        best_score = float('-inf') 
        
        legal_moves = game.get_legal_moves()                
        if legal_moves:
            best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        else:
            return game.utility(self), best_move            
                 
        if depth <= 0:
            return self.score(game, self), best_move
               
        for move in legal_moves:
            new_score, _ = self.alphabeta_min(game.forecast_move(move), depth - 1, alpha, beta)
            if (new_score >= best_score):
                best_score = new_score
                best_move = move
            
            if (best_score >= beta):
                return best_score, best_move
           
            alpha = max(alpha, best_score)
                
        return best_score, best_move
    
    def alphabeta_min(self, game, depth, alpha, beta):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        else:
            return game.utility(self), best_move
            
        
        if depth <= 0:
            return self.score(game, self), best_move

        best_score = float('inf')        
        for move in legal_moves:
            new_score, _ = self.alphabeta_max(game.forecast_move(move), depth - 1, alpha, beta)
            if (new_score <= best_score):
                best_score = new_score
                best_move = move
             
            if (best_score <= alpha):
                return best_score, best_move
            
            beta = min(beta, best_score)
            
        return best_score, best_move