"""Release entry point for the six AdaGCN/AdaIGCN citation experiments.

This remains a TensorFlow 1.x graph-mode trainer. The release protocol keeps
target ground truth outside training: target ``group`` is loaded only after
the best source-validation checkpoint has been restored.
"""

from __future__ import division, print_function

import csv
import gc
import os
import random
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf


RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from models import GCN  # noqa: E402
from utils import (  # noqa: E402
    construct_feed_dict,
    construct_feed_dict_source,
    construct_feed_dict_target,
    load_source_citation_data,
    load_target_citation_graph,
    load_target_citation_labels,
    preprocess_adj,
    preprocess_features,
)


tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


# Release/protocol flags.
flags.DEFINE_string('source', 'citationv1', 'Source citation dataset name.')
flags.DEFINE_string('target', 'dblpv7', 'Target citation dataset name.')
flags.DEFINE_boolean('single_pair', True, 'Run only --source to --target.')
flags.DEFINE_string('data_root', 'Citation_mat', 'Directory containing citation .mat files.')
flags.DEFINE_integer('seed', 1234, 'Python/NumPy/TensorFlow training seed.')
flags.DEFINE_integer('split_seed', 42, 'Independent source 80/10/10 split seed.')
flags.DEFINE_string('result_csv', '', 'Optional append-only final-result CSV.')
flags.DEFINE_boolean('use_becat', False, 'Enable the full TensorFlow BECAT adapter.')
flags.DEFINE_boolean(
    'use_geometry_adapter', False,
    'Deprecated alias for --use_becat; retained for command compatibility.'
)

# Original AdaGCN/AdaIGCN settings. Defaults are preserved, not retuned.
flags.DEFINE_integer('epochs', 1000, 'Number of training epochs.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate.')
flags.DEFINE_float('l2_param', 5e-5, 'L2 weight.')
flags.DEFINE_integer('signal', 1, 'Release path supports unlabeled-target DA only (1).')
flags.DEFINE_float('da_param', 1.0, 'Wasserstein domain-loss weight.')
flags.DEFINE_float('gp_param', 10.0, 'Gradient-penalty weight.')
flags.DEFINE_integer('D_train_step', 10, 'Discriminator steps per epoch.')
flags.DEFINE_float('shrinking', 0.8, 'Learning-rate shrink factor.')
flags.DEFINE_string('hiddens_gcn', '1000|100|16', 'GCN hidden dimensions.')
flags.DEFINE_string('hiddens_clf', '', 'Classifier hidden dimensions.')
flags.DEFINE_string('hiddens_dis', '16', 'Discriminator hidden dimensions.')
flags.DEFINE_string('da_method', 'WD', 'Domain adaptation method.')
flags.DEFINE_float('lr_gen', 1.5e-3, 'Generator learning rate.')
flags.DEFINE_float('lr_dis', 1.5e-3, 'Discriminator learning rate.')
flags.DEFINE_boolean('with_metrics', True, 'Build TensorFlow F1 metrics.')
flags.DEFINE_integer('num_gcn_layers', 1, 'Number of GCN layers.')
flags.DEFINE_integer('smoothing_steps', 10, 'IGCN smoothing steps.')
flags.DEFINE_string('gnn', 'igcn', 'Convolution type: gcn or igcn.')

# Historical internal component flags remain for compatibility, but the public
# main-table interface is only --use_becat.
flags.DEFINE_boolean('use_latent_recon', False, 'Compatibility flag.')
flags.DEFINE_boolean('use_soft_bec_loss', False, 'Compatibility flag.')
flags.DEFINE_boolean('use_latent_topk_bec', False, 'Internal latent BEC branch.')
flags.DEFINE_integer('latent_topk_k', 10, 'Top-k latent neighbors.')
flags.DEFINE_float('latent_topk_tau', 0.5, 'Latent top-k temperature.')
flags.DEFINE_float('lambda_latent_bec', 0.0001, 'Legacy trainer BECAT loss weight.')
flags.DEFINE_boolean('latent_bec_use_mmd', False, 'Include RBF-MMD alignment.')
flags.DEFINE_float('latent_bec_N', -1.0, 'Bakry-Emery dimension; -1 is infinity.')
flags.DEFINE_integer('latent_bec_sample_size', 128, 'BECAT sample size.')
flags.DEFINE_float('latent_bec_mmd_sigma', 1.0, 'MMD kernel sigma.')
flags.DEFINE_float('lambda_latent_bec_stat', 0.1, 'Legacy statistic-loss weight.')
flags.DEFINE_integer('latent_bec_warmup_epochs', 100, 'BECAT loss warm-up epochs.')
flags.DEFINE_integer('latent_bec_update_interval', 20, 'BECAT loss update interval.')
flags.DEFINE_boolean('debug_latent_bec_shapes', False, 'Print internal shape diagnostics.')
flags.DEFINE_boolean('use_sparse_gt_head', False, 'Internal sparse GT head branch.')
flags.DEFINE_integer('sparse_gt_num_heads', 2, 'Sparse GT attention heads.')
flags.DEFINE_integer('sparse_gt_num_layers', 1, 'Sparse GT layers.')
flags.DEFINE_float('sparse_gt_dropout', 0.3, 'Sparse GT dropout.')
flags.DEFINE_boolean('sparse_gt_use_curv_bias', False, 'Use curvature attention bias.')
flags.DEFINE_float('sparse_gt_curv_bias_beta', 0.05, 'Curvature bias strength.')
flags.DEFINE_enum('sparse_gt_curv_bias_mode', 'abs', ['abs', 'hybrid'], 'Bias mode.')
flags.DEFINE_integer('sparse_gt_fourier_m', 4, 'Fourier curvature frequencies.')
flags.DEFINE_float('sparse_gt_fourier_sigma', 1.0, 'Fourier frequency stddev.')
flags.DEFINE_integer('sparse_gt_fourier_seed', 1234, 'Fixed Fourier feature seed.')
flags.DEFINE_float('sparse_gt_eta_abs', 1.0, 'Absolute curvature-bias weight.')
flags.DEFINE_float('sparse_gt_eta_fourier', 0.05, 'Fourier curvature-bias weight.')
flags.DEFINE_enum(
    'sparse_gt_bias_norm', 'scale', ['none', 'scale', 'zscore'], 'Bias normalization.'
)
flags.DEFINE_boolean(
    'sparse_gt_to_critic', False,
    'Internal experiment; main-table reproduction requires false.'
)
flags.DEFINE_integer('sparse_gt_attention_k', -1, 'Sparse GT neighbor count.')
flags.DEFINE_integer(
    'sparse_gt_curv_bias_warmup', -1,
    'Curvature-bias warm-up; negative reuses BECAT warm-up.'
)
flags.DEFINE_boolean('sparse_gt_use_self_loop', True, 'Add sparse GT self-loops.')
flags.DEFINE_float('sparse_gt_residual_alpha', 0.1, 'Sparse GT residual scale.')
flags.DEFINE_boolean('sparse_gt_use_layernorm', False, 'Apply sparse GT layer norm.')


CANONICAL_TASKS = [
    ('A_to_C', 'acmv9', 'citationv1'),
    ('A_to_D', 'acmv9', 'dblpv7'),
    ('C_to_A', 'citationv1', 'acmv9'),
    ('C_to_D', 'citationv1', 'dblpv7'),
    ('D_to_A', 'dblpv7', 'acmv9'),
    ('D_to_C', 'dblpv7', 'citationv1'),
]
DATASETS = set(name for _, source, target in CANONICAL_TASKS for name in (source, target))
FINAL_RESULT_COLUMNS = [
    'source', 'target', 'seed', 'split_seed', 'best_epoch',
    'best_source_val_accuracy', 'source_val_accuracy', 'source_test_accuracy',
    'target_accuracy', 'target_micro_f1', 'target_macro_f1',
    'target_weighted_f1', 'use_becat'
]


def _resolve_release_flags():
    FLAGS.use_geometry_adapter = bool(FLAGS.use_becat or FLAGS.use_geometry_adapter)
    FLAGS.use_becat = FLAGS.use_geometry_adapter
    if FLAGS.signal != 1:
        raise ValueError('The citation main-table release supports only --signal=1.')
    if FLAGS.da_method != 'WD':
        raise ValueError('The release preserves the original WD adaptation path.')
    if FLAGS.sparse_gt_to_critic:
        raise ValueError('Main-table reproduction requires --sparse_gt_to_critic=false.')
    if not FLAGS.with_metrics:
        raise ValueError('Final Micro-F1/Macro-F1 reporting requires --with_metrics=true.')
    if not FLAGS.use_becat and (FLAGS.use_latent_topk_bec or FLAGS.use_sparse_gt_head):
        raise ValueError('Partial geometry branches are not public main-table methods.')


def _set_training_seed(seed):
    """Seed each newly reset TF graph and the training RNGs."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.compat.v1.set_random_seed(int(seed))


def _is_latent_bec_epoch(epoch):
    if not (FLAGS.use_becat or FLAGS.use_latent_topk_bec):
        return False
    interval = max(int(FLAGS.latent_bec_update_interval), 1)
    warmup = int(FLAGS.latent_bec_warmup_epochs)
    if epoch < warmup:
        return False
    return (epoch - warmup) % interval == 0


def _resolved_curvature_warmup():
    warmup = int(FLAGS.sparse_gt_curv_bias_warmup)
    if warmup < 0:
        warmup = int(FLAGS.latent_bec_warmup_epochs)
    return warmup


def _is_sparse_gt_curv_bias_epoch(epoch):
    if FLAGS.use_becat:
        return epoch >= _resolved_curvature_warmup()
    if not FLAGS.use_sparse_gt_head or not FLAGS.sparse_gt_use_curv_bias:
        return False
    return epoch >= _resolved_curvature_warmup()


def _zero_geometry_metrics():
    return {
        'latent_bec_loss': 0.0,
        'latent_bec_mmd': 0.0,
        'latent_bec_stat': 0.0,
        'latent_kappa_src_mean': 0.0,
        'latent_kappa_tgt_mean': 0.0,
    }


def _geometry_fetches(model):
    return [
        model.latent_bec_loss,
        model.latent_bec_mmd,
        model.latent_bec_stat,
        model.latent_kappa_src_mean,
        model.latent_kappa_tgt_mean,
    ]


def _geometry_metrics(values):
    names = [
        'latent_bec_loss', 'latent_bec_mmd', 'latent_bec_stat',
        'latent_kappa_src_mean', 'latent_kappa_tgt_mean'
    ]
    return dict((name, float(value)) for name, value in zip(names, values))


def _evaluation_flags(feed_dict, curvature_bias_active):
    feed_dict.update({
        PLACEHOLDERS['dropout']: 0.0,
        PLACEHOLDERS['sparse_gt_dropout']: 0.0,
        PLACEHOLDERS['sparse_gt_curv_bias_active']: curvature_bias_active,
        PLACEHOLDERS['geometry_adapter_training']: False,
    })
    return feed_dict


def evaluate_source(sess, model, features, support, labels, mask,
                    curvature_bias_active):
    feed_dict = construct_feed_dict_source(
        features, support, labels, mask, PLACEHOLDERS
    )
    _evaluation_flags(feed_dict, curvature_bias_active)
    values = sess.run(
        [model.clf_loss_s, model.accuracy_s, model.micro_f1_s,
         model.macro_f1_s, model.weighted_f1_s],
        feed_dict=feed_dict
    )
    return {
        'loss': float(values[0]),
        'accuracy': float(values[1]),
        'micro_f1': float(values[2]),
        'macro_f1': float(values[3]),
        'weighted_f1': float(values[4]),
    }


def evaluate_target(sess, model, features, support, labels, mask,
                    curvature_bias_active):
    feed_dict = construct_feed_dict_target(
        features, support, labels, mask, PLACEHOLDERS
    )
    _evaluation_flags(feed_dict, curvature_bias_active)
    values = sess.run(
        [model.clf_loss_t, model.accuracy_t, model.micro_f1_t,
         model.macro_f1_t, model.weighted_f1_t],
        feed_dict=feed_dict
    )
    return {
        'loss': float(values[0]),
        'accuracy': float(values[1]),
        'micro_f1': float(values[2]),
        'macro_f1': float(values[3]),
        'weighted_f1': float(values[4]),
    }


def _make_placeholders(source_features, target_features, num_classes):
    num_source = source_features[2][0]
    num_target = target_features[2][0]
    return {
        'support_t': [tf.compat.v1.sparse_placeholder(tf.float32)],
        'features_t': tf.compat.v1.sparse_placeholder(tf.float32, shape=target_features[2]),
        'labels_t': tf.compat.v1.placeholder(tf.float32, shape=(num_target, num_classes)),
        'labels_mask_t': tf.compat.v1.placeholder(tf.bool, shape=(num_target,)),
        'num_features_nonzero_t': tf.compat.v1.placeholder(tf.int32),
        'support_s': [tf.compat.v1.sparse_placeholder(tf.float32)],
        'features_s': tf.compat.v1.sparse_placeholder(tf.float32, shape=source_features[2]),
        'labels_s': tf.compat.v1.placeholder(tf.float32, shape=(num_source, num_classes)),
        'labels_mask_s': tf.compat.v1.placeholder(tf.bool, shape=(num_source,)),
        'num_features_nonzero_s': tf.compat.v1.placeholder(tf.int32),
        'dropout': tf.compat.v1.placeholder_with_default(0.0, shape=()),
        'lr_dis': tf.compat.v1.placeholder(tf.float32, shape=()),
        'lr_gen': tf.compat.v1.placeholder(tf.float32, shape=()),
        'latent_bec_active': tf.compat.v1.placeholder_with_default(False, shape=()),
        'sparse_gt_dropout': tf.compat.v1.placeholder_with_default(0.0, shape=()),
        'sparse_gt_curv_bias_active': tf.compat.v1.placeholder_with_default(False, shape=()),
        'geometry_adapter_training': tf.compat.v1.placeholder_with_default(False, shape=()),
        'l': tf.compat.v1.placeholder(tf.float32, shape=()),
    }


def _format_final_result(result):
    values = ['FINAL_RESULT']
    for name in FINAL_RESULT_COLUMNS:
        value = result[name]
        if isinstance(value, float):
            value = '{:.6f}'.format(value)
        values.append('{}={}'.format(name, value))
    return ','.join(values)


def _append_result_csv(result):
    if not FLAGS.result_csv:
        return
    write_header = not os.path.exists(FLAGS.result_csv) or os.path.getsize(FLAGS.result_csv) == 0
    with open(FLAGS.result_csv, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FINAL_RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(dict((name, result[name]) for name in FINAL_RESULT_COLUMNS))


def train_one_pair(source_name, target_name, source_data, target_data,
                   target_label_path):
    source_features, source_support, source_labels, source_train_labels = source_data[:4]
    train_mask, val_mask, test_mask = source_data[4:]
    target_features, target_support = target_data
    num_source = source_features[2][0]
    num_target = target_features[2][0]
    num_classes = source_labels.shape[1]

    dummy_target_labels = np.zeros((num_target, num_classes), dtype=source_labels.dtype)
    dummy_target_mask = np.zeros(num_target, dtype=bool)
    model = GCN(
        PLACEHOLDERS, target_features[2][1], num_source, num_target, logging=True
    )
    saver = tf.compat.v1.train.Saver(
        var_list=tf.compat.v1.global_variables(), max_to_keep=1
    )

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    best_source_val_accuracy = -np.inf
    best_epoch = 0
    best_curvature_bias_active = False
    lr_gen = float(FLAGS.lr_gen)
    lr_dis = float(FLAGS.lr_dis)
    started = time.time()

    if FLAGS.use_becat:
        cfg = model.geometry_adapter_config
        if not (
            cfg.use_latent_topk_bec
            and cfg.use_sparse_gt_head
            and cfg.sparse_gt_use_curv_bias
        ):
            raise ValueError('--use_becat requires the full BECAT geometry adapter.')
        print(
            'BECAT_CONFIG,k={},tau={},lambda={},stat={},warmup={},interval={},'
            'to_critic={}'.format(
                cfg.latent_topk_k, cfg.latent_topk_tau, cfg.lambda_latent_bec,
                cfg.lambda_latent_bec_stat, FLAGS.latent_bec_warmup_epochs,
                FLAGS.latent_bec_update_interval, cfg.sparse_gt_to_critic
            )
        )

    with tempfile.TemporaryDirectory(prefix='adagcn_best_') as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

        for epoch in range(FLAGS.epochs):
            if (epoch + 1) >= 500 and (epoch + 1) % 100 == 0:
                lr_gen *= FLAGS.shrinking
                lr_dis *= FLAGS.shrinking

            latent_bec_active = _is_latent_bec_epoch(epoch)
            curvature_bias_active = _is_sparse_gt_curv_bias_epoch(epoch)
            feed_dict = construct_feed_dict(
                target_features, target_support,
                dummy_target_labels, dummy_target_mask,
                source_features, source_support,
                source_train_labels, train_mask,
                PLACEHOLDERS, lr_gen, lr_dis
            )
            feed_dict.update({
                PLACEHOLDERS['dropout']: FLAGS.dropout,
                PLACEHOLDERS['latent_bec_active']: latent_bec_active,
                PLACEHOLDERS['sparse_gt_dropout']: FLAGS.sparse_gt_dropout,
                PLACEHOLDERS['sparse_gt_curv_bias_active']: curvature_bias_active,
                PLACEHOLDERS['geometry_adapter_training']: True,
            })

            wd_loss = 0.0
            dis_loss = 0.0
            for _ in range(FLAGS.D_train_step):
                _, wd_loss, dis_loss = sess.run(
                    [model.wd_d_op, model.wd_loss, model.dis_loss_total],
                    feed_dict=feed_dict
                )

            geometry = _zero_geometry_metrics()
            if latent_bec_active:
                outputs = sess.run(
                    [model.opt_op_total_s, model.clf_loss_s] + _geometry_fetches(model),
                    feed_dict=feed_dict
                )
                source_train_loss = float(outputs[1])
                geometry = _geometry_metrics(outputs[2:])
            else:
                _, source_train_loss = sess.run(
                    [model.opt_op_total_s_base, model.clf_loss_s], feed_dict=feed_dict
                )
                source_train_loss = float(source_train_loss)

            source_val = evaluate_source(
                sess, model, source_features, source_support, source_labels,
                val_mask, curvature_bias_active
            )
            if source_val['accuracy'] > best_source_val_accuracy:
                best_source_val_accuracy = source_val['accuracy']
                best_epoch = epoch + 1
                best_curvature_bias_active = curvature_bias_active
                saver.save(sess, checkpoint_path)

            print(
                'TRAIN_EPOCH,epoch={},source={},target={},source_train_loss={:.6f},'
                'source_val_accuracy={:.6f},best_source_val_accuracy={:.6f},'
                'wd_loss={:.6f},dis_loss={:.6f},becat_active={},'
                'becat_loss={:.6f}'.format(
                    epoch + 1, source_name, target_name, source_train_loss,
                    source_val['accuracy'], best_source_val_accuracy,
                    float(wd_loss), float(dis_loss), latent_bec_active,
                    geometry['latent_bec_loss']
                )
            )

        if best_epoch == 0:
            raise RuntimeError('No source-validation checkpoint was created.')

        saver.restore(sess, checkpoint_path)
        print(
            'CHECKPOINT_RESTORED,best_epoch={},best_source_val_accuracy={:.6f}'.format(
                best_epoch, best_source_val_accuracy
            )
        )

        source_val = evaluate_source(
            sess, model, source_features, source_support, source_labels,
            val_mask, best_curvature_bias_active
        )
        source_test = evaluate_source(
            sess, model, source_features, source_support, source_labels,
            test_mask, best_curvature_bias_active
        )

        # Protocol boundary: target ground truth first enters memory here, after
        # the source-validation checkpoint has been restored.
        target_labels = load_target_citation_labels(
            target_label_path, num_target, num_classes
        )
        print('TARGET_LABELS_LOADED,phase=final_evaluation,target={}'.format(target_name))
        target_mask = np.ones(num_target, dtype=bool)
        target_eval = evaluate_target(
            sess, model, target_features, target_support, target_labels,
            target_mask, best_curvature_bias_active
        )

    result = {
        'source': source_name,
        'target': target_name,
        'seed': int(FLAGS.seed),
        'split_seed': int(FLAGS.split_seed),
        'best_epoch': int(best_epoch),
        'best_source_val_accuracy': float(best_source_val_accuracy),
        'source_val_accuracy': source_val['accuracy'],
        'source_test_accuracy': source_test['accuracy'],
        'target_accuracy': target_eval['accuracy'],
        'target_micro_f1': target_eval['micro_f1'],
        'target_macro_f1': target_eval['macro_f1'],
        'target_weighted_f1': target_eval['weighted_f1'],
        'use_becat': bool(FLAGS.use_becat),
    }
    print(_format_final_result(result))
    print('RUN_DURATION_SECONDS,{:.3f}'.format(time.time() - started))
    _append_result_csv(result)
    sess.close()
    return result


def _selected_tasks():
    if FLAGS.single_pair:
        if FLAGS.source not in DATASETS or FLAGS.target not in DATASETS:
            raise ValueError('Unknown citation source or target dataset.')
        if FLAGS.source == FLAGS.target:
            raise ValueError('Source and target must differ.')
        return [('single_pair', FLAGS.source, FLAGS.target)]
    return list(CANONICAL_TASKS)


def run_pair(task_name, source_name, target_name):
    tf.compat.v1.reset_default_graph()
    _set_training_seed(FLAGS.seed)

    source_path = os.path.join(FLAGS.data_root, '{}.mat'.format(source_name))
    target_path = os.path.join(FLAGS.data_root, '{}.mat'.format(target_name))
    source_adj, source_x, source_y, source_y_train, train_mask, val_mask, test_mask = (
        load_source_citation_data(source_path, FLAGS.split_seed)
    )
    target_adj, target_x = load_target_citation_graph(target_path)

    source_features = preprocess_features(source_x)
    target_features = preprocess_features(target_x)
    source_support = [preprocess_adj(source_adj)]
    target_support = [preprocess_adj(target_adj)]

    global PLACEHOLDERS
    PLACEHOLDERS = _make_placeholders(
        source_features, target_features, source_y.shape[1]
    )
    print(
        'RUN_START,task={},source={},target={},seed={},split_seed={},use_becat={}'.format(
            task_name, source_name, target_name, FLAGS.seed, FLAGS.split_seed,
            FLAGS.use_becat
        )
    )
    print(
        'SOURCE_SPLIT_COUNTS,train={},val={},test={},total={}'.format(
            int(np.sum(train_mask)), int(np.sum(val_mask)), int(np.sum(test_mask)),
            source_y.shape[0]
        )
    )
    result = train_one_pair(
        source_name,
        target_name,
        (source_features, source_support, source_y, source_y_train,
         train_mask, val_mask, test_mask),
        (target_features, target_support),
        target_path
    )

    del source_adj, source_x, source_y, source_y_train
    del target_adj, target_x, source_features, target_features
    gc.collect()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    return result


def main(_):
    _resolve_release_flags()
    results = []
    for task_name, source_name, target_name in _selected_tasks():
        results.append(run_pair(task_name, source_name, target_name))
    if len(results) > 1:
        print('FINAL_SUMMARY,count={}'.format(len(results)))
        for result in results:
            print(_format_final_result(result))


PLACEHOLDERS = None


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main)
