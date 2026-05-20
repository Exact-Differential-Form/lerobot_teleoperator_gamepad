from .compat import register_third_party_devices
from .record_manual import record_trossen_gamepad_manual


def main():
    register_third_party_devices()
    record_trossen_gamepad_manual()


if __name__ == "__main__":
    main()
