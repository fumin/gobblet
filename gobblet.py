import numpy as np
import pyspiel


_NUM_SIZES = 3
_PIECES_PER_SIZE = 2
_BOARD_WIDTH = 3
_BOARD_SIZE = _BOARD_WIDTH * _BOARD_WIDTH
_GAME_TYPE = pyspiel.GameType(
        short_name="gobblet",
        long_name="gobblet, a capturing variant of tic-tac-toe",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=2,
        min_num_players=2,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
        parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
        num_distinct_actions=(_NUM_SIZES+_BOARD_SIZE)*_BOARD_SIZE,
        max_chance_outcomes=0,
        num_players=2,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=1000)


class Game(pyspiel.Game):
    """A Gobblet game."""

    def __init__(self, params={}):
        super().__init__(_GAME_TYPE, _GAME_INFO, params)

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return State(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return Observer(self)


class State(pyspiel.State):
    """A state of a Gobblet game."""

    def __init__(self, game):
        super().__init__(game)
        self._cur_player = 0
        self._player0_score = 0.0
        self._is_terminal = False

        # Reserves tracks the number of pieces for each size.
        self.reserves = np.ones([game.num_players(), _NUM_SIZES], dtype=int)
        self.reserves *= _PIECES_PER_SIZE

        # Board tracks the pieces for each size on each location on the board.
        self.board = np.ones([_BOARD_WIDTH, _BOARD_WIDTH, _NUM_SIZES], dtype=int)
        self.board *= pyspiel.PlayerId.INVALID

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        legals = []

        # Actions that place a new piece on the board.
        for src_size, pieces in enumerate(self.reserves[player]):
            if pieces == 0:
                continue
            for dsty in range(self.board.shape[0]):
                for dstx in range(self.board.shape[1]):
                    dst = (dsty, dstx)
                    dst_piece = _largest_piece(self.board[dst])
                    if src_size > dst_piece.size:
                        legals.append(Action(reserves=src_size, dst=dst).idx())

        # Actions that move an existing piece on the board.
        for srcy in range(self.board.shape[0]):
            for srcx in range(self.board.shape[1]):
                src = (srcy, srcx)
                src_piece = _largest_piece(self.board[src])
                if src_piece.player != player:
                    continue
                for dsty in range(self.board.shape[0]):
                    for dstx in range(self.board.shape[1]):
                        dst = (dsty, dstx)
                        dst_piece = _largest_piece(self.board[dst])
                        if src_piece.size > dst_piece.size:
                            legals.append(Action(src=src, dst=dst).idx())

        return legals

    def _apply_action(self, action_idx):
        action = _action_from_idx(action_idx)
        if action.reserves != -1:
            self.reserves[self._cur_player, action.reserves] -= 1
            self.board[action.dst][action.reserves] = self._cur_player
        else:
            size = _largest_piece(self.board[action.src]).size
            self.board[action.src][size] = pyspiel.PlayerId.INVALID
            self.board[action.dst][size] = self._cur_player

        if _line_player(self.board) != pyspiel.PlayerId.INVALID:
            self._is_terminal = True
            self._player0_score = 1.0 if self._cur_player == 0 else -1.0
        else:
            self._cur_player = 1 - self._cur_player

    def _action_to_string(self, player, action_idx):
        action = _action_from_idx(action_idx)
        return str(action)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]


class Observer:
    """Observer, conforming to the PyObserver interface."""

    def __init__(self, game):
        ps = game.num_players()
        pieces = [
                ["player", -1, [1]],
                ["board", -1, [_BOARD_WIDTH, _BOARD_WIDTH, _NUM_SIZES, ps]],
                ]
        for i in range(len(pieces)):
            pieces[i][1] = np.prod(pieces[i][2])

        # Build the single flat tensor.
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, dtype=int)

        # Build the named & reshaped views of the bits of the flat tensor.
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
          self.dict[name] = self.tensor[index:index + size].reshape(shape)
          index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        if state.current_player() == 1:
            self.dict["player"][0] = 1
        for y in range(state.board.shape[0]):
            for x in range(state.board.shape[1]):
                for size in range(state.board.shape[2]):
                    p = state.board[y, x, size]
                    if p != pyspiel.PlayerId.INVALID:
                        self.dict["board"][y, x, size, p] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        rs_str = str(state.reserves)
        board_str = _board_to_string(state.board)
        return "reserves:\n"+rs_str+"\nboard:\n"+board_str


_ACTION_OFFSET = _NUM_SIZES * _BOARD_SIZE


class Action:

    def __init__(self, reserves=-1, src=(-1, -1), dst=(-1, -1)):
        self.reserves = reserves
        self.src = src
        self.dst = dst

    def idx(self):
        src = self.src[0]*_BOARD_WIDTH + self.src[1]
        dst = self.dst[0]*_BOARD_WIDTH + self.dst[1]
        if self.reserves != -1:
            return self.reserves*_BOARD_SIZE + dst
        return _ACTION_OFFSET + src*_BOARD_SIZE + dst

    def __str__(self):
        return "reserves {} src {} dst {}".format(self.reserves, self.src, self.dst)


def _action_from_idx(idx):
    if idx < _ACTION_OFFSET:
        dst_idx = idx % _BOARD_SIZE
        dst = (int(dst_idx / _BOARD_WIDTH), int(dst_idx % _BOARD_WIDTH))
        return Action(reserves=int(idx/_BOARD_SIZE), dst=dst)

    idx -= _ACTION_OFFSET
    src_idx = idx / _BOARD_SIZE
    src = (int(src_idx / _BOARD_WIDTH), int(src_idx % _BOARD_WIDTH))
    dst_idx = idx % _BOARD_SIZE
    dst = (int(dst_idx / _BOARD_WIDTH), int(dst_idx % _BOARD_WIDTH))
    return Action(src=src, dst=dst)


def _line_player(board):
    for row in range(board.shape[0]):
        player = _pieces_player(board[row, :])
        if player != pyspiel.PlayerId.INVALID:
            return player
    for col in range(board.shape[1]):
        player = _pieces_player(board[:, col])
        if player != pyspiel.PlayerId.INVALID:
            return player

    diag = []
    for y in range(board.shape[0]):
        x = board.shape[0]-1 - y
        diag.append(board[y, x])
    player = _pieces_player(diag)
    if player != pyspiel.PlayerId.INVALID:
        return player

    negdiag = []
    for y in range(board.shape[0]):
        negdiag.append(board[y, y])
    player = _pieces_player(negdiag)
    if player != pyspiel.PlayerId.INVALID:
        return player

    return pyspiel.PlayerId.INVALID


def _pieces_player(towers):
    player = _largest_piece(towers[0]).player
    for tw in towers[1:]:
        if _largest_piece(tw).player != player:
            return pyspiel.PlayerId.INVALID
    return player


class Piece:

    def __init__(self, player, size):
        self.player = player
        self.size = size


def _largest_piece(tower):
    for size in range(len(tower)-1, -1, -1):
        player = tower[size]
        if player != pyspiel.PlayerId.INVALID:
            return Piece(player, size)
    return Piece(pyspiel.PlayerId.INVALID, -1)


def _board_to_string(board):
    row_strs = []
    for y in range(board.shape[0]):
        rstr = ["" for _ in range(_NUM_SIZES)]
        for x in range(board.shape[1]):
            for size, player in enumerate(board[y, x]):
                if player == 0:
                    rstr[size] += "o "
                elif player == 1:
                    rstr[size] += " x"
                else:
                    rstr[size] += "  "
            if x != board.shape[1]-1:
                for size in range(len(rstr)):
                    rstr[size] += "|"
        row_strs.append("\n".join(rstr) + "\n")

    row_delim = ("-" * (_BOARD_WIDTH*3-1)) + "\n"
    board_str = row_delim.join(row_strs)
    # Remove redundant ending newline.
    board_str = board_str[:-1]

    return board_str


pyspiel.register_game(_GAME_TYPE, Game)
