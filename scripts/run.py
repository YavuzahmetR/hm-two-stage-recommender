"""Run a project command without depending on the terminal's working directory."""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    'inspect_data': 'hm_recommender.data.inspect_data',
    'ranking_dataset': 'hm_recommender.data.ranking_dataset',
    'build_v6': 'hm_recommender.data.build_v6',
    'prepare_two_tower': 'hm_recommender.data.prepare_two_tower',
    'popularity_baseline': 'hm_recommender.baselines.popularity_baseline',
    'repeat_baseline': 'hm_recommender.baselines.repeat_baseline',
    'covisitation': 'hm_recommender.candidates.covisitation',
    'train_ranker': 'hm_recommender.ranking.train_ranker',
    'train_multiweek': 'hm_recommender.ranking.train_multiweek',
    'train_v6': 'hm_recommender.ranking.train_v6',
    'tune_ranker': 'hm_recommender.ranking.tune_ranker',
    'integrate_two_tower': 'hm_recommender.ranking.integrate_two_tower',
    'train_two_tower': 'hm_recommender.retrieval.train_two_tower',
    'experiment_two_tower': 'hm_recommender.retrieval.experiment_two_tower',
    'feature_two_tower': 'hm_recommender.retrieval.feature_two_tower',
    'compare_history': 'hm_recommender.evaluation.compare_history',
    'compare_sources': 'hm_recommender.evaluation.compare_sources',
    'evaluate_report': 'hm_recommender.evaluation.evaluate_report',
    'evaluate_two_tower': 'hm_recommender.evaluation.evaluate_two_tower',
    'inspect_tuned_ranker': 'hm_recommender.evaluation.inspect_tuned_ranker',
    'finish_project': 'hm_recommender.evaluation.finish_project',
    'archive_results': 'hm_recommender.evaluation.archive_results',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)
    sys.argv = [args.command, *args.arguments]
    module = importlib.import_module(COMMANDS[args.command])
    module.main()


if __name__ == "__main__":
    main()
