import numpy as np
import torch
from IPython.display import HTML, display
from torch.utils.tensorboard import SummaryWriter

from tic_tac_toe.Player import Player
from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY


def print_board(board):
    display(HTML("""
    <style>
    .rendered_html table, .rendered_html th, .rendered_html tr, .rendered_html td {
      border: 1px  black solid !important;
      color: black !important;
    }
    </style>
    """ + board.html_str()))


def play_game(board: Board, player1: Player, player2: Player):
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

    # noinspection PyUnboundLocalVariable
    player1.final_result(final_result)
    # noinspection PyUnboundLocalVariable
    player2.final_result(final_result)
    return final_result


def board_state_to_one_hot_nn_input(state: np.ndarray, device: torch.device, side: int) -> torch.Tensor:
    # Convert numpy state directly to tensor once
    t_state = torch.as_tensor(state, device=device)

    other_side = Board.other_side(side)

    # Create masks directly in PyTorch
    is_me = (t_state == side).float()
    is_other = (t_state == other_side).float()
    is_empty = (t_state == EMPTY).float()

    # Concatenate into the expected (27,) shape
    return torch.cat([is_me, is_other, is_empty])

def board_state_to_cnn_input(state: np.ndarray, device: torch.device, side: int) -> torch.Tensor:
    # 1. Convert numpy state to tensor and reshape to the grid dimensions (3x3)
    # If your board is already 3x3, this just ensures the shape is correct.
    t_state = torch.as_tensor(state, device=device).view(3, 3)

    other_side = Board.other_side(side)

    # 2. Create masks (Spatial Features)
    # Each of these will be a (3, 3) matrix
    is_me = (t_state == side).float()
    is_other = (t_state == other_side).float()
    is_empty = (t_state == EMPTY).float()

    # 3. Stack into (Channels, Height, Width) -> (3, 3, 3)
    # Using stack with dim=0 creates a new dimension at the start
    cnn_input = torch.stack([is_me, is_other, is_empty], dim=0)

    return cnn_input
def evaluate_batch(player1: Player, player2: Player, num_games: int = 100,
                   silent: bool = False, writer: SummaryWriter = None, epoch: int = 0):
    board = Board()
    draw_count = 0
    cross_count = 0
    naught_count = 0
    for _ in range(num_games):
        result = play_game(board, player1, player2)
        if result == GameResult.CROSS_WIN:
            cross_count += 1
        elif result == GameResult.NAUGHT_WIN:
            naught_count += 1
        else:
            draw_count += 1

    # Calculate percentages
    p1_win_pct = cross_count / num_games
    p2_win_pct = naught_count / num_games
    draw_pct = draw_count / num_games

    if writer:
        writer.add_scalar(f"{player1.name}/1st/Win_Rate", p1_win_pct, epoch)
        writer.add_scalar(f"{player1.name}/1st/Loss_Rate", p2_win_pct, epoch)
        writer.add_scalar(f"{player1.name}/1st/DRAW_Rate", draw_pct, epoch)

        writer.add_scalar(f"{player2.name}/2nd/Loss_Rate", p1_win_pct, epoch)
        writer.add_scalar(f"{player2.name}/2nd/Win_Rate", p2_win_pct, epoch)
        writer.add_scalar(f"{player2.name}/2nd/DRAW_Rate", draw_pct, epoch)

    if not silent:
        print("After {} game we have draws: {}, Player 1 wins: {}, and Player 2 wins: {}.".format(num_games, draw_count,
                                                                                                  cross_count,
                                                                                                  naught_count))

        print("Which gives percentages of draws: {:.2%}, Player 1 wins: {:.2%}, and Player 2 wins:  {:.2%}".format(
            draw_count / num_games, cross_count / num_games, naught_count / num_games))

    return cross_count, naught_count, draw_count


def evaluate_players(p1: Player, p2: Player, games_per_evaluation_batch=100, num_evaluation_batches=100,
                     writer=None, silent: bool = False):
    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    for i in range(num_evaluation_batches):
        p1win, p2win, draw = evaluate_batch(p1, p2, games_per_evaluation_batch, silent)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter = game_counter + 1
        game_number.append(game_counter)

    return game_number, p1_wins, p2_wins, draws
