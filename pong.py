import retro        # pip install gym-retro
import numpy as np  # pip install numpy
#import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle
import os
import multiprocessing
import cv2
import time

env = retro.make(game='Pong-Atari2600')



def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)


    env.reset()
    ob, _, _, _ = env.step(env.action_space.sample())
    inx = int(ob.shape[0]/8)
    iny = int(ob.shape[1]/8)
    fitnesses = []


    score1=0
    score2=0
    # Run the given simulation for up to num_steps time steps.
    fitness = 0.0
    done = False
    start_time=time.time()
    series_of_actions=[]
    while not done:
        env.render()


        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))
        imgarray = np.ndarray.flatten(ob)
        imgarray = np.interp(imgarray, (0, 254), (-1, +1))
        nnOut = net.activate(imgarray)
        keys=[]


        for i in nnOut:
            if i > 0.5:
                keys.append(1)
            else:
                keys.append(0)

        actions=[0]*4+keys+[0]*2
        series_of_actions.append(keys)


        ob, rew, done, info = env.step(actions)

        score1=info['score1']
        score2=info['score2']


        if score1 >19 or score2 >19:
            done = True
    print(series_of_actions)
    run_time=time.time()-start_time

    fitness=score2-score1/(run_time-2)
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pong_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(10, eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
