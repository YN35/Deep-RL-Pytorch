import time
import yaml
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from mllib.counter import EpsCounter
from utils import load_config_and_make_output_directory, load_main_model, load_models


def train(cfg):
    models = load_models(cfg)
    main_model = load_main_model(cfg, models)

    cnt = EpsCounter(cfg.ls)
    writer = SummaryWriter(cfg.misc.output_dir)

    t0 = time.perf_counter()
    while True:
        # Evaluate the current model.
        if cnt.do_eval():
            dpts_curr = cnt.incr_eval()
            result = main_model.eval_epsd()
            
            print(f'[Eval] {dpts_curr}pnts ({time.perf_counter() - t0:.0f}s)')
            for env_name, env_result in result.items():
                print(f'env: {env_name}')
                print(yaml.safe_dump(env_result['stat'], default_flow_style=False))
                main_model.eval_result_writer(env_name, env_result, dpts_curr, writer)

        # Save the current model snapshot.
        if cnt.do_snapshot() or (main_model.finish_training() or cnt.finish_training()):
            dpts_curr = cnt.incr_snapshot()
            snapshot_name_prefix = f'{dpts_curr:016d}'
            main_model.save_state_dict(snapshot_name_prefix)
            print(f'[Snapshot] {snapshot_name_prefix}')

        # Finish the training?
        if main_model.finish_training() or cnt.finish_training():
            print(f'[Finish]')
            break

        # Step training.
        result = main_model.train_epsd()
        cnt.incr_train_stat(result)

        # Log train stat.
        if cnt.do_train_log():
            dpts_curr, loss, loss_terms = cnt.incr_train_log()
            lr = main_model.get_lr()

            print(f'[Train] {dpts_curr}dpts ({time.perf_counter() - t0:.0f}s) lr: {lr:.3e} loss: {loss:.3e}')
            print(' '.join([f'{k}: {v:.3e}' for k, v in loss_terms.items()]))

            writer.add_scalar('train/lr', lr, dpts_curr)
            writer.add_scalar('train/loss', loss, dpts_curr)
            for k, v in loss_terms.items():
                writer.add_scalar(f'loss terms/{k}', v, dpts_curr)
                

        # Update lr.
        if cnt.do_update_lr():
            dpts_curr, gain = cnt.incr_update_lr()
            if gain is not None:
                gain_smth = main_model.update_lr(gain)
                print(f'per sample improvement: {gain_smth:.3e}')
                writer.add_scalar('train/gain smoothed', gain_smth, dpts_curr)


if __name__ == '__main__':
    train(load_config_and_make_output_directory(default_config='./config/train.yaml', output_dir='./runs_train'))