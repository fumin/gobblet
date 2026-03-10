import logging
logging.basicConfig()
lg = logging.getLogger()
[lg.removeHandler(h) for h in lg.handlers]
lg.addHandler(logging.StreamHandler())
lg.setLevel(logging.INFO)
lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))


import pyspiel

import gobblet


def main():
    game = pyspiel.load_game("gobblet")
    actions = [
            gobblet.Action(reserves=0, dst=(1, 1)),
            gobblet.Action(reserves=2, dst=(1, 1)),
            gobblet.Action(reserves=2, dst=(2, 2)),
            ]

    state = game.new_initial_state()
    for act in actions:
        infoset_str = state.information_state_string()
        mask = state.legal_actions_mask()
        logging.info("current_player %d", state.current_player())
        logging.info("%s", infoset_str)
        logging.info("infoset %s", state.information_state_tensor())
        logging.info("act %s %d", act, act.idx())
        if mask[act.idx()] != 1:
            raise Exception("invalid action")
        state.apply_action(act.idx())

    if state.is_terminal():
        logging.info("terminal state")
    else:
        logging.info("current_player %d", state.current_player())
        infoset_str = state.information_state_string()
        logging.info("%s", infoset_str)
    logging.info("returns %s", state.returns())


if __name__ == "__main__":
    main()
