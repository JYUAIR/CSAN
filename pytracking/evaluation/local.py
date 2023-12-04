from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/opt/data/private/houyueen/TransT/ltr/workspace/checkpoints/ltr/transt/transt'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/opt/data/private/houyueen/mydata/pytracking/pytracking/pytracking/otb'
    settings.result_plot_path = '/opt/data/private/houyueen/TransT/pytracking/result_plots/'
    settings.results_path = '/opt/data/private/houyueen/TransT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/opt/data/private/houyueen/TransT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = '/opt/data/private/houyueen/mydata/pytracking/pytracking/pytracking/vot'
    settings.youtubevos_dir = ''

    return settings

