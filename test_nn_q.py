import torch
from torch.utils.tensorboard import SummaryWriter

from tic_tac_toe import RndMinMaxAgent
from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY
from tic_tac_toe.RandomPlayer import RandomPlayer
from util import print_board, play_game, evaluate_batch
# from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
# from tic_tac_toe.TabularQPlayer import TQPlayer
from tic_tac_toe.SimpleNNQPlayer import NNQPlayer
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

writer = SummaryWriter('runs/tic_tac_toe_experiment_1')

board = Board()
nnplayer = NNQPlayer("QLearner1", device= device, writer=writer)
nnplayer2 = NNQPlayer("QLearner2", device= device, writer=writer)

deep_nnplayer = NNQPlayer("DeepQLearner1", device= device, writer=writer)

rndplayer = RandomPlayer()
mm_player = RndMinMaxAgent()
# tq_player = TQPlayer()

p1_wins = []
p2_wins = []
draws = []
game_number = []
game_counter = 0

num_evaluation_batchs = 10
games_per_evaluation_batch = 100
num_training_evaluation_batchs = 100

# nnplayer rndplayer mm_player
p2_t = nnplayer
p1_t = nnplayer2

p1 = p1_t
p2 = p2_t

for i in range(num_training_evaluation_batchs):
    p1win, p2win, draw = evaluate_batch(p1_t, p2_t, games_per_evaluation_batch, False
                                        , writer=writer, epoch=i)
    p1_t.log_weights()
    p2_t.log_weights()

    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    game_counter = game_counter + 1
    game_number.append(game_counter)

writer.close()

nnplayer.training= False
nnplayer2.training= False

for i in range(num_evaluation_batchs):
    p1win, p2win, draw = evaluate_batch(p1, p2, games_per_evaluation_batch, False)
    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    game_counter = game_counter + 1
    game_number.append(game_counter)

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
