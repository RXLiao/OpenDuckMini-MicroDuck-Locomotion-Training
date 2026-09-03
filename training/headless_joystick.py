"""Open Duck Mini locomotion with a 10-D leg-only PPO action space.

The four head joints remain in every observation.  They are driven from the
head command (the last four command entries), exactly like deployment, rather
than being produced by the PPO actor.
"""

import jax.numpy as jp

from .joystick import Joystick, default_config


HEAD_ACTION_IDS = (5, 6, 7, 8)
LEG_ACTION_IDS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)


class HeadlessJoystick(Joystick):
    """Joystick environment exposing only the ten locomotion actuators."""

    @property
    def action_size(self) -> int:
        return len(LEG_ACTION_IDS)

    def step(self, state, leg_action):
        if leg_action.shape[-1] != len(LEG_ACTION_IDS):
            raise ValueError(
                f"expected {len(LEG_ACTION_IDS)} leg actions, got {leg_action.shape}"
            )

        # Joystick.step advances the command ramp before stepping physics.  Use
        # that same next ramp value here so externally-controlled head targets
        # and the command stored in the resulting observation stay aligned.
        next_ramp_step = jp.minimum(state.info["command_ramp_step"] + 1, 50)
        ramp = next_ramp_step / 50.0
        head_command = ramp * state.info["desired_command"][3:7]

        full_action = jp.zeros(self._actuators)
        full_action = full_action.at[jp.array(LEG_ACTION_IDS)].set(leg_action)
        # Joystick converts normalized actions to targets as
        # default + action * action_scale.  Dividing here therefore produces
        # exactly default_head + head_command, with no PPO head contribution.
        full_action = full_action.at[jp.array(HEAD_ACTION_IDS)].set(
            head_command / self._config.action_scale
        )
        return super().step(state, full_action)

