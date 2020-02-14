from collections import namedtuple

import numpy as np
# from pysc2.env.environment import TimeStep, StepType
# from pysc2.lib import actions
# from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType


def log_transform(x, scale):
    # 8 is a "feel good" magic number and doesn't mean anything here.
    return np.log(8 * x / scale + 1)


def get_visibility_flag(visibility_feature):
    # 0=hidden, 1=fogged, 2=visible
    return np.expand_dims(visibility_feature == 2, axis=0)


# def numeric_idx_and_scale(set):
#     idx_and_scale = [
#         (k.index, k.scale) for k in set
#         if k.type == FeatureType.SCALAR
#     ]
#     idx, scale = [np.array(k) for k in zip(*idx_and_scale)]
#     scale = scale.reshape(-1, 1, 1)
#     # [k.name for k in SCREEN_FEATURES if k.type == FeatureType.SCALAR]
#     return idx, scale


def stack_list_of_dicts(d):
    return {key: np.stack([a[key] for a in d]) for key in d[0]}


# def get_available_actions_flags(obs):
#     # Get the available actions
#     avilable_actions_dense = np.zeros(len(actions.FUNCTIONS), dtype=np.float32)
#     avilable_actions_dense[obs['available_actions']] = 1
#     return avilable_actions_dense


class ObsProcesser:
    N_SCREEN_CHANNELS = 13
    N_MINIMAP_CHANNELS = 5

    # def __init__(self):
    #     self.screen_numeric_idx, self.screen_numeric_scale = \
    #         numeric_idx_and_scale(SCREEN_FEATURES)
    #     self.minimap_numeric_idx, self.minimap_numeric_scale = \
    #         numeric_idx_and_scale(MINIMAP_FEATURES)
    #
    #     screen_flag_names = ["creep", "power", "selected"]
    #     self.screen_flag_idx = [k.index for k in SCREEN_FEATURES
    #         if k.name in screen_flag_names]
    #
    #     minimap_flag_names = ["creep", "camera", "selected"]
    #     self.mimimap_flag_idx = [k.index for k in MINIMAP_FEATURES
    #         if k.name in minimap_flag_names]
    #
    # def get_screen_numeric(self, obs):
    #     screen_obs = np.array(obs["feature_screen"])
    #     scaled_scalar_obs = log_transform(
    #         screen_obs[self.screen_numeric_idx], self.screen_numeric_scale
    #     )
    #     return np.r_[
    #         scaled_scalar_obs,
    #         screen_obs[self.screen_flag_idx],
    #         get_visibility_flag(screen_obs[SCREEN_FEATURES.visibility_map.index])
    #     ]
    #
    # def get_mimimap_numeric(self, obs):
    #     minimap_obs = np.array(obs["feature_minimap"])
    #
    #     # This is only height_map for minimiap
    #     scaled_scalar_obs = log_transform(
    #         minimap_obs[self.minimap_numeric_idx], self.minimap_numeric_scale
    #     )
    #     return np.r_[
    #         scaled_scalar_obs,
    #         minimap_obs[self.mimimap_flag_idx],
    #         get_visibility_flag(minimap_obs[MINIMAP_FEATURES.visibility_map.index])
    #     ]

    # def process_one_input(self, timestep: TimeStep): # TimeStep is for SC2
    def process_one_input(self, timestep):
        # obs = {}
        # obs['observation']={}
        # obs['observation']['rgb_screen'] = timestep
        #obs = timestep.observation
        #print(obs.rgb_screen)
        #feature_screen = np.copy(obs["feature_screen"]) # THIS FIXES EVERYTHING AND YOU CAN SEE THE feature_screen obs cauz of the error for max_recursions etc
        #feature_minimap = np.copy(obs["feature_minimap"])
        # pp_obs = {
        #     FEATURE_KEYS.rgb_screen: timestep
        # }
        pp_obs = {
            FEATURE_KEYS.rgb_screen: timestep,#['img'],
            # FEATURE_KEYS.alt_view: timestep['nextstepimage'],
            # FEATURE_KEYS.altitudes: timestep['altitude'],
            # FEATURE_KEYS.image_vol: timestep['image_volume'],
            # FEATURE_KEYS.joined: timestep['joined'],
        }

        # pp_obs = {
        #     FEATURE_KEYS.player_relative_screen: np.array(obs.rgb_screen),
        #     FEATURE_KEYS.player_relative_minimap: np.array(obs.rgb_minimap)
        # }

        return pp_obs

    def process(self, obs_list):
        """
        :param obs_list: list[TimeStep],
        :return: dict of key -> array[env, ...]

        screen features are originally NCHW,
        however tf on CPU works only with NHWC so returning those here
        """
        # TODO: vectorize this function and class
        pp_obs = [self.process_one_input(obs) for obs in obs_list] # obs will contain n_envs obs. Each obs will contain whatever the env outs but you want to group same outs together.
        pp_obs = stack_list_of_dicts(pp_obs) # group tgther same stuff

        # for k in ["screen_numeric", "minimap_numeric"]:
        #     pp_obs[k] = np.transpose(pp_obs[k], [0, 2, 3, 1])

        return pp_obs

    def combine_batch(self, mb_obs):
        return stack_list_of_dicts(mb_obs)


def make_default_args(arg_names):
    default_args = []
    spatial_seen = False
    spatial_arguments = ["screen", "minimap", "screen2"]
    for k in arg_names:
        if k in spatial_arguments:
            spatial_seen = True
            continue
        else:
            assert not spatial_seen, "got %s argument after spatial argument" % k
            default_args.append([0])

    return tuple(default_args), spatial_seen


def convert_point_to_rectangle(point, delta, dim):
    def l(x):
        return max(0, min(x, dim - 1))

    p1 = [l(k - delta) for k in point]
    p2 = [l(k + delta) for k in point]
    return p1, p2


# def arg_names():
#     x = [[a.name for a in k.args] for k in actions.FUNCTIONS]
#     assert all("minimap2" not in k for k in x)
#     return x


# def find_rect_function_id():
#     """
#     this is just a change-safe way to return 3
#     """
#     x = [k.id for k, names in zip(actions.FUNCTIONS, arg_names()) if "screen2" in names]
#     assert len(x) == 1
#     return x[0]


class ActionProcesser:
    def __init__(self, dim, rect_delta=5):
        # self.default_args, is_spatial = zip(*[make_default_args(k) for k in arg_names()])
        # self.is_spatial = np.array(is_spatial)
        # self.rect_select_action_id = find_rect_function_id()
        self.rect_delta = rect_delta
        self.dim = dim

    # def make_one_action(self, action_id, spatial_coordinates):
    #     # (MINE) Maybe put an argument for enabling Queued actions?
    #     args = list(self.default_args[action_id])
    #     assert all(s < self.dim for s in spatial_coordinates)
    #     if action_id == self.rect_select_action_id:
    #         args.extend(convert_point_to_rectangle(spatial_coordinates, self.rect_delta, self.dim))
    #     elif self.is_spatial[action_id]:
    #         # NOTE: in pysc2 v 1.2 the action space (x,y) is flipped. Handling that conversion here
    #         # in all other places we operate with the "non-flipped" coordinates
    #         args.append(spatial_coordinates[::-1])
    #
    #     return actions.FunctionCall(action_id, args)

    # def process(self, action_ids, spatial_action_2ds):
    #     return [self.make_one_action(a_id, coord)
    #         for a_id, coord in zip(action_ids, spatial_action_2ds)]

    def combine_batch(self, mb_actions):
        d = {}
        d[FEATURE_KEYS.selected_action_id] = np.stack(k for k in mb_actions)
        #d[FEATURE_KEYS.selected_spatial_action] = np.stack(k[1] for k in mb_actions)
        #d[FEATURE_KEYS.is_spatial_action_available] = self.is_spatial[ d[FEATURE_KEYS.selected_action_id] ]
        return d

    # def combine_recurrent_batch(self, mb_actions):
    #     d = {}
    #     d[FEATURE_KEYS.selected_action_id] = np.stack(k for k in mb_actions).transpose()
    #     #d[FEATURE_KEYS.selected_spatial_action] = np.stack(k[1] for k in mb_actions)
    #     #d[FEATURE_KEYS.is_spatial_action_available] = self.is_spatial[ d[FEATURE_KEYS.selected_action_id] ]
    #     return d


FEATURE_LIST = (
    "alt0_grass",
    "alt0_bush",
    "alt0_drone",
    "alt0_hiker",
    "alt1_pine",
    "alt1_pines",
    "alt1_drone",
    "alt2_drone",
    "alt3_drone",
    'alt_view',
    "minimap_numeric",
    "screen_numeric",
    "screen_unit_type",
    "player_relative_screen",
    "player_relative_minimap",
    "rgb_screen",
    "is_spatial_action_available",
    "selected_spatial_action",
    "selected_action_id",
    # "available_action_ids",
    "value_target",
    "value_target_goal",
    "value_target_fire",
    "advantage",
    "prev_actions",
    "prev_rewards",
    "altitudes",
    "image_vol",
    "joined",
    "actup_probs",

)

AgentInputTuple = namedtuple("AgentInputTuple", FEATURE_LIST)
FEATURE_KEYS = AgentInputTuple(*FEATURE_LIST)
