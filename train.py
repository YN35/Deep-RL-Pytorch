

from utils import load_config, load_main_model, load_models, set_device, load_envs


def train(cfg):
    cfg = set_device(cfg)
    cfg = load_models(cfg)
    cfg = load_envs(cfg)
    main_model = load_main_model(cfg)
    print('[Start]')

    while True:
        # Step training.
        main_model.train_epsd()
        
        main_model.add_epsd()
        # Finish the training?
        if main_model.finish_training() or main_model.epsd_finish_training():
            print(f'[Finish]')
            break


if __name__ == '__main__':
    train(load_config())