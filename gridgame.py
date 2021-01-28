from common.maps import *
import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from envs.core import *
import PIL
from PIL import Image


from flask import Flask, render_template, request, send_from_directory
import queue
import random
import threading


app = Flask(__name__)

pending_choice = None
choice_queue = queue.Queue()
svg_queue = queue.Queue()
game_thread = None
game_initated = False

size_factor = 30

class Game(threading.Thread):
    def __init__(self,env):
        super().__init__(daemon=True)
        self.env = env


    def run(self):
        obs = self.env._gridmap_to_image()
        done = False
        while True:
            choice = choice_queue.get()
            if choice:
                if done and choice == 'reset':
                    obs = self.env.reset()
                    done = False
                elif choice in ['up','down','left','right', 'noop']:
                    obs, r, done, info = self.env.step([],choice)
            svg_queue.put(obs)


            # if choice:
            #     obs, r, done, info = self.env.step([],choice)
            #     if done:
            #         if choice == 'reset':
            #             obs = self.env.reset()
            # svg_queue.put(obs)



            # choice = choice_queue.get()
            # if choice:
            #     print('pre-obs choice', choice)
            #     obs, r, done, info = self.env.step([])
            # svg_queue.put(obs)



@app.route('/images/<path:path>')
def images(path):
    print(path)
    return send_from_directory('images/', path)


@app.route("/")
def play_game():
    global game_thread
    global game_initated
    if not game_initated:
        env = envs.generic_env.GenericEnv(dims=(10,10),choice_queue=choice_queue)
        goal = Goal(env,entity_type='goal',color='green')
        player = WebAgent(env,entity_type='agent',color='orange')
        predator = ChasingBlockingAdvisary(env,entity_type='adversary',color='red',obs_type='data',position='near-goal')
        env.reset()
    game_initated = True
    if not game_thread:
        game_thread = Game(env)

        game_thread.start()
    choice_queue.put(request.args.get("choice", ""))
    return render_template("game.html", svg=svg_queue.get())

if __name__ == '__main__':
    app.run(threaded=True)