from common.maps import *
import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from envs.core import *
import PIL
from PIL import Image
import sqlite3 as sql
import time

from flask import Flask, render_template, request, send_from_directory, abort, session, redirect, url_for
import jinja2
import queue
import random
import pickle
import datetime as dt

application = app = Flask(__name__)


app.secret_key = 'hello'
mission_configuration = pickle.load(open('100position.netw','rb'))
environments = {}
bad_configs = [0, 3, 9, 11, 13, 14, 15, 20, 23, 36, 39, 43, 44, 47, 48,52,55,61]

with sql.connect("player_info.db") as con:
    cursor = con.cursor()
    sqltxt = '''CREATE TABLE IF NOT EXISTS player_data (
            id TEXT PRIMARY KEY,
            userid TEXT,
            episode INTEGER,
            step INTEGER,
            reward INTEGER,
            goal_x INTEGER,
            goal_y INTEGER,
            player_x INTEGER,
            player_y INTEGER,
            predator_x INTEGER,
            predator_y INTEGER,
            goal_distance INTEGER,
            adversary_distance INTEGER,
            action INTEGER,
            actionTime INTEGER);'''
    cursor.execute(sqltxt)
    sqltxt = '''CREATE TABLE IF NOT EXISTS player_buttons (
            id TEXT PRIMARY KEY,
            userid TEXT,
            participate INTEGER,
            instruct_next INTEGER,
            turnbased_next INTEGER,
            movement_next INTEGER,
            winning_next INTEGER,
            losing_next INTEGER);'''
    cursor.execute(sqltxt)

@app.route('/game')
def game():
    global environments
    userid = session['user']

    # print("session", session)
    # print("user-s", userid)
    yoursteps = 0
    beststeps = 0
    if not userid in environments:
        #userid = userid
        #session["user"] = userid
        environments[userid] = {}
        environments[userid]['env'] = envs.generic_env.GenericEnv(dims=(10, 10))
        environments[userid]['goal'] = Goal(environments[userid]['env'], entity_type='goal', color='green')
        environments[userid]['player'] = WebAgent(environments[userid]['env'], entity_type='agent', color='orange')
        environments[userid]['predator'] = ChasingBlockingAdvisary(environments[userid]['env'], entity_type='adversary', color='red', obs_type='data', position='near-goal')
        environments[userid]['done'] = False
        environments[userid]['reward'] = 0
        environments[userid]['step'] = 0
        environments[userid]['episode'] = 0
        environments[userid]['configuration'] = 0
        obs = environments[userid]['env'].reset(config=mission_configuration[environments[userid]['configuration']])
        session['obs'] = obs

    else:
        if environments[userid]['episode'] >= 50:
            return redirect(url_for("complete"))
        obs = session['obs']
    beststeps = mission_configuration[environments[userid]['configuration']]['steps']
    #print(mission_configuration[environments[userid]['episode']])
    episode = environments[userid]['episode']
    return render_template('game.html', svg=obs, turn_number=0, yoursteps=yoursteps,episodes=episode,beststeps=beststeps)

@app.route('/')
def root():
    return redirect(url_for("setup"))

@app.route('/instructions', methods=["POST", "GET"])
def instructions():
    userid = session['user']

    instruct_next, turnbased_next, movement_next, winning_next, losing_next, participate = 0, 0, 0, 0, 0, 0
    if request.method == "POST":
        with sql.connect("player_info.db") as con:
            instruct_next = dt.datetime.now()
            sqltxt = '''INSERT INTO player_buttons(userid,participate,instruct_next,turnbased_next,movement_next,winning_next,losing_next) VALUES (?,?,?,?,?,?,?)'''
            cur = con.cursor()
            cur.execute(sqltxt, (
            userid, participate, instruct_next, turnbased_next, movement_next, winning_next, losing_next))
            con.commit()
            return redirect(url_for("turn_based"))
    return render_template("instructions.html")

@app.route('/turn_based', methods=["POST", "GET"])
def turn_based():
    userid = session['user']
    instruct_next, turnbased_next, movement_next, winning_next, losing_next, participate = 0, 0, 0, 0, 0, 0
    if request.method == "POST":
        with sql.connect("player_info.db") as con:
            turnbased_next = dt.datetime.now()
            sqltxt = '''INSERT INTO player_buttons(userid,participate, instruct_next,turnbased_next,movement_next,winning_next,losing_next) VALUES (?,?,?,?,?,?,?)'''
            cur = con.cursor()
            cur.execute(sqltxt, (
                userid, participate, instruct_next, turnbased_next, movement_next, winning_next, losing_next))
            con.commit()
            return redirect(url_for("movement"))
    return render_template("turn_based.html")

@app.route("/movement", methods=["POST", "GET"])
def movement():
    userid = session['user']
    instruct_next, turnbased_next, movement_next, winning_next, losing_next, participate = 0, 0, 0, 0, 0, 0
    if request.method == "POST":
        with sql.connect("player_info.db") as con:
            movement_next = dt.datetime.now()
            sqltxt = '''INSERT INTO player_buttons(userid,participate,instruct_next,turnbased_next,movement_next,winning_next,losing_next) VALUES (?,?,?,?,?,?,?)'''
            cur = con.cursor()
            cur.execute(sqltxt, (
                userid, participate, instruct_next, turnbased_next, movement_next, winning_next, losing_next))
            con.commit()
            return redirect(url_for("winning"))
    return render_template("movement.html")

@app.route('/losing', methods=["POST", "GET"])
def losing():
    userid = session['user']
    instruct_next, turnbased_next, movement_next, winning_next, losing_next, participate = 0, 0, 0, 0, 0, 0
    if request.method == "POST":
        with sql.connect("player_info.db") as con:
            losing_next = dt.datetime.now()
            sqltxt = '''INSERT INTO player_buttons(userid,participate,instruct_next,turnbased_next,movement_next,winning_next,losing_next) VALUES (?,?,?,?,?,?,?)'''
            cur = con.cursor()
            cur.execute(sqltxt, (
                userid, participate, instruct_next, turnbased_next, movement_next, winning_next, losing_next))
            con.commit()
            return redirect(url_for("game"))
    return render_template("losing.html")

@app.route('/winning', methods=["POST", "GET"])
def winning():
    userid = session['user']
    instruct_next, turnbased_next, movement_next, winning_next, losing_next, participate = 0, 0, 0, 0, 0, 0
    if request.method == "POST":
        with sql.connect("player_info.db") as con:
            winning_next = dt.datetime.now()
            sqltxt = '''INSERT INTO player_buttons(userid,participate,instruct_next,turnbased_next,movement_next,winning_next,losing_next) VALUES (?,?,?,?,?,?,?)'''
            cur = con.cursor()
            cur.execute(sqltxt, (
                userid, participate, instruct_next, turnbased_next, movement_next, winning_next, losing_next))
            con.commit()
            return redirect(url_for("losing"))
    return render_template("winning.html")

@app.route('/consent', methods=["POST","GET"])
def consent():
    userid = session['user']
    instruct_next, turnbased_next, movement_next, winning_next, losing_next, participate = 0, 0, 0, 0, 0, 0
    if request.method == "POST":
        with sql.connect("player_info.db") as con:
            participate = dt.datetime.now()
            sqltxt = '''INSERT INTO player_buttons(userid,participate,instruct_next,turnbased_next,movement_next,winning_next,losing_next) VALUES (?,?,?,?,?,?,?)'''
            cur = con.cursor()
            cur.execute(sqltxt, (
                userid, participate, instruct_next, turnbased_next, movement_next, winning_next, losing_next))
            con.commit()

            return redirect(url_for("instructions"))
    else:
        return render_template("consent.html")


@app.route('/complete', methods=["POST","GET"])
def complete():
    global environments
    userid = session['user']
    if request.method == "GET":
        if userid not in environments:
            return redirect(url_for("instructions"))
        if environments[userid]['episode'] < 50:
            return redirect(url_for("instructions"))
    #Code generation here.
    return render_template('complete.html')

@app.route('/setup', methods=["POST", "GET"])
def setup():

    if request.method == "POST":
        userid = request.form["code"]
        print("USER:",userid)
        session['user'] = userid
        return redirect(url_for("consent"))
    else:
        return render_template('setup.html')

@app.route('/images/<path:path>')
def images(path):
    print(path)
    return send_from_directory('images/', path)

def scrub(table_name):
    return ''.join(chr for chr in table_name if chr.isalnum())

@app.route('/move', methods = ['POST'])
def move():
    global environments
    global bad_configs
    data = request.form

    userid = session['user']
    choice = int(data['choice'])
    turn_number = int(request.form["step_number"])
    #turn_number = data['step_number']
    #print("CHOICE",request.form.getlist())


    # userprint("envos",environments)
    # print('useer',userid)
    done = None
    r = 0
    txt = ''
    step = environments[userid]['step']
    observation = environments[userid]['env'].current_grid_map.copy()

    goal_location = np.where(observation == 2)
    player_location = np.where(observation == 3)
    predator_location = np.where(observation == 4)
    if player_location[0].size == 0 or goal_location[0].size == 0 or predator_location[0].size == 0:
        goal_distance = -1
        adversary_distance = -1
        player_location = (-1,-1)
        goal_location = (-1,-1)
        predator_location = (-1,-1)
        action = choice
        actiontime = dt.datetime.now()
    else:
        # import pdb;
        # pdb.set_trace()
        player_location = (int(player_location[0]), int(player_location[1]))
        goal_location = (int(goal_location[0]), int(goal_location[1]))
        predator_location = (int(predator_location[0]), int(predator_location[1]))
        # print(goal_location, player_location, predator_location)
        action = choice
        actiontime = dt.datetime.now()
        env = environments[userid]['env']
        # print('observation:')
        # print(np.array2string(observation))
        # print("callling GPT 1")

        path_to_goal = env.getPathTo(player_location, goal_location, free_spaces=env.free_spaces)
        # print(np.array2string(path_to_goal))
        points_in_path = np.where(path_to_goal == -1)
        points_in_path = list(zip(points_in_path[0], points_in_path[1]))
        goal_distance = len(points_in_path)
        # print("calling GPT 2")
        path_to_adversary = env.getPathTo((player_location[0], player_location[1]),
                                          (predator_location[0], predator_location[1]), free_spaces=env.free_spaces)
        points_in_path = np.where(path_to_adversary == -1)
        points_in_path = list(zip(points_in_path[0], points_in_path[1]))
        adversary_distance = len(points_in_path)
    # print(goal_distance, adversary_distance, "!!!!!!")
    #print(np.array2string(observation))
    #print(np.where(observation==4))
    episode = environments[userid]['episode']

    if choice == 99 and environments[userid]['done']:
        print("1",choice, environments[userid]['done'])
        configuration = environments[userid]['configuration']
        if environments[userid]['reward'] == -1:
            obs = environments[userid]['env'].reset(config=mission_configuration[configuration])
            environments[userid]['done'] = False
            environments[userid]['reward'] = 0
            environments[userid]['step'] = 0
            environments[userid]['episode'] = environments[userid]['episode']
            environments[userid]['configuration'] = configuration
        else:
            configuration += 1
            while configuration in bad_configs:
                configuration += 1
            print("configuration", configuration)
            obs = environments[userid]['env'].reset(config=mission_configuration[configuration])
            environments[userid]['done'] = False
            environments[userid]['reward'] = 0
            environments[userid]['step'] = 0
            environments[userid]['episode'] = environments[userid]['episode'] + 1
            environments[userid]['configuration'] = configuration
            #environments[userid]['gridmap'] = environments[userid]['env'].current_grid_map
            session['obs'] = obs
        if environments[userid]['episode'] >=50:
            return redirect(url_for("complete"))
    elif choice == 99 and not environments[userid]['done']:
        print("2",choice,environments[userid]['done'])
        obs = session['obs']
    elif not choice == 99:
        print("3",data['choice'])
        obs, r, done, info = environments[userid]['env'].step([],data['choice'])
        environments[userid]['done'] = done
        environments[userid]['reward'] = r
        environments[userid]['step'] = environments[userid]['step'] + 1
        session['obs'] = obs

    if not done == None and not r == None:
        if done == True and r == 1:
            txt = f'<p style=\"font-size:35px; color:red; position:absolute; top:50px; left:25px;\">You win</p><p style=\"font-size:25px; color:red; position:absolute; top:100px; left:25px;\">Press reset to move on</p>'
            done == None
        if done == True and r == -1:
            txt = f'<p style=\"font-size:35px; color:red; position:absolute; top:50px; left:25px;\">You lose</p><p style=\"font-size:25px; color:red; position:absolute; top:100px; left:25px;\">Press reset to try again!</p>'
            done == None

    with sql.connect("player_info.db") as con:
        # goal_location = np.where(observation == 2)
        goal_x = int(goal_location[0])
        goal_y = int(goal_location[1])
        # player_location = np.where(observation == 3)
        #player_location = list(zip(player_location[0],player_location[1]))[0]
        # print("PLAYER", player_location)
        if len(player_location) == 0:
        #if not len(player_location):
            player_x = -1
            player_y = -1
        else:
            player_x = int(player_location[0])
            player_y = int(player_location[1])
        # predator_location = np.where(observation == 4)
        #predator_location = list(zip(predator_location[0],predator_location[1]))[0]
        #if not len(predator_location):
        if len(predator_location) == 0:
            predator_x = -1
            predator_y = -1
        else:
            predator_x = int(predator_location[0])
            predator_y = int(predator_location[1])




        # agent_location = np.where(self.env.current_grid_map == agent)
        # path = self.env.getPathTo((my_location[0], my_location[1]), (agent_location[0], agent_location[1]),
        #                           free_spaces=self.env.free_spaces)
        # points_in_path = np.where(path == -1)
        # points_in_path = list(zip(points_in_path[0], points_in_path[1]))
        # distance_to_agents[agent] = {'dist': len(points_in_path), 'path': path}

        sqltxt = '''INSERT INTO player_data(userid,episode,step,reward,goal_x,goal_y,player_x,player_y,predator_x,predator_y,goal_distance,adversary_distance,action,actionTime) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        cur = con.cursor()
        cur.execute(sqltxt,(userid,episode,step,r,goal_x,goal_y,player_x,player_y,predator_x,predator_y,goal_distance,adversary_distance,action,actiontime))
        con.commit()

    episode = environments[userid]['episode']
    return render_template('game.html', svg=obs, turn_number=turn_number+1,yoursteps=environments[userid]['step'],beststeps=mission_configuration[environments[userid]['configuration']]['steps'],episodes=episode,feedback=txt)



if __name__ == '__main__':

    app.debug =  True
    app.run(host='0.0.0.0', port=8080)