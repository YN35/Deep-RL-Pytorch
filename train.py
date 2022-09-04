

from utils import load_config, load_main_model, load_models


def train(cfg):
    models = load_models(cfg)
    main_model = load_main_model(cfg, models)

    while True:
        # Step training.
        main_model.train_epsd()
        
        # Finish the training?
        if main_model.finish_training() or main_model.epsd_finish_training():
            print(f'[Finish]')
            break


if __name__ == '__main__':
    train(load_config())