"""Run the standard ONNX evaluator with the 10-action headless environment."""

import eval_onnx
from playground.open_duck_mini_v2 import headless_joystick


# eval_onnx constructs joystick.Joystick by name.  The headless module also
# imports the 14-action base class under that name, so explicitly bind the
# evaluator-facing symbol to the 10-action subclass.
headless_joystick.Joystick = headless_joystick.HeadlessJoystick
eval_onnx.joystick = headless_joystick


if __name__ == "__main__":
    eval_onnx.main()
