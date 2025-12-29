import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from tic_tac_toe import RndMinMaxAgent
from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY
from tic_tac_toe.DQNPlayer import DQNPlayer
from tic_tac_toe.DoubleDQNPlayer import DoubleDQNPlayer
from tic_tac_toe.DuelingDoubleDQNPlayer import DuelingDoubleDQNPlayer
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.ReplayNNQPlayer import ReplayNNQPlayer
from util import print_board, play_game, evaluate_batch
# from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
# from tic_tac_toe.TabularQPlayer import TQPlayer
from tic_tac_toe.SimpleNNQPlayer import NNQPlayer
from tic_tac_toe.EGreedyNNQPlayer import EGreedyNNQPlayer
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

board = Board()

rndplayer = RandomPlayer("RandomPlayer")
rnd_mm_player = RndMinMaxAgent("RandomMinMaxPlayer")
mm_player = MinMaxAgent("MinMaxPlayer")
# tq_player = TQPlayer()

nnegreedy_player = EGreedyNNQPlayer("EGreedyNNQPlayer", device= device)
nnplayer = NNQPlayer("NNQPlayer", device= device)
dqn_player = DQNPlayer("DQNPlayer", device= device)
replayNNQPlayer = ReplayNNQPlayer("ReplayNNQPlayer", device= device)
double_dqn_player = DoubleDQNPlayer("DoubleDQNPlayer", device= device)
dueling_double_player = DuelingDoubleDQNPlayer("DuelingDoubleDQNPlayer", device= device)


# nnplayer2 = NNQPlayer("NNQPlayer2", device= device, writer=writer)
#


p1_wins = []
p2_wins = []
draws = []
game_number = []
game_counter = 0

games_per_training_batch = 120
num_training_batches = 200
num_training_eval_games = 50

num_evaluation_batches = 2
games_per_evaluation_batch = 100

# nnplayer rndplayer mm_player
p2 = dueling_double_player
p1 = rnd_mm_player

# Define a descriptive name for the current experiment
experiment_name = f"{p1.name}_vs_{p2.name}"
# Generate a unique timestamp string
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Combine them to create the unique log directory path
log_dir_path = os.path.join("runs", f"{timestamp}_{experiment_name}")
writer = SummaryWriter(log_dir_path)
p1.writer = writer
p2.writer = writer

p2.log_graph()
# Initialize the SummaryWriter with the specific path

for i in range(num_training_batches):
    # Training phase
    p1.training = True
    p2.training = True
    p1win, p2win, draw = evaluate_batch(p1, p2, games_per_training_batch, False, writer=writer, epoch=i)
    p1_wins.append(p1win/games_per_training_batch)
    p2_wins.append(p2win/games_per_training_batch)
    draws.append(draw/games_per_training_batch)
    p1.log_weights()
    p2.log_weights()

    # Evaluation phase
    # p1.training = False
    # p2.training = False
    # p1win, p2win, draw = evaluate_batch(p1, p2, num_training_eval_games, False, writer=writer, epoch=i)
    # p1_wins.append(p1win/num_training_eval_games)
    # p2_wins.append(p2win/num_training_eval_games)
    # draws.append(draw/num_training_eval_games)

    game_counter = game_counter + 1
    game_number.append(game_counter)

writer.close()

p1.training= False
p2.training= False

# p1 = nnegreedy_player
# p2 = mm_player

for i in range(num_evaluation_batches):
    p1win, p2win, draw = evaluate_batch(p1, p2, games_per_evaluation_batch, False)
    p1_wins.append(p1win/games_per_evaluation_batch)
    p2_wins.append(p2win/games_per_evaluation_batch)
    draws.append(draw/games_per_evaluation_batch)
    game_counter = game_counter + 1
    game_number.append(game_counter)

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
