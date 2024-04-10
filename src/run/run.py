# current_sensitivity = float(metrics_exp["Sensitivity"])
# current_specificity = float(metrics_exp["Specificity"])
# current_AUC = float(metrics_exp["AUC"])
# current_F1 = float(metrics_exp["f1_score"])
# current_ACC = float(metrics_exp["acc"])
#
# history_per_mask['sensitivity'].append(current_sensitivity)
# history_per_mask['specificity'].append(current_specificity)
# history_per_mask['AUC'].append(current_AUC)
# history_per_mask['F1'].append(current_F1)
# history_per_mask['ACC'].append(current_ACC)
#
# if current_AUC > h_param['AUC']['AUC']:
#     h_param['AUC']['ACC'] = current_ACC
#     h_param['AUC']['AUC'] = current_AUC
#     h_param['AUC']['F1'] = current_F1
#     h_param['AUC']['dilation_radius'] = int(
#         mask.split('_')[1])
#     h_param['AUC']['p_center_distance'] = int(
#         mask.split('_')[2])
#     h_param['AUC']['Otsu'] = int(mask.split('_')[3])
#     h_param['AUC']['n_segments'] = n_segments
#     h_param['AUC']['compactness'] = compactness
#     h_param['AUC']['p_seeds_final'] = p_seeds_final
#
# if current_F1 > h_param['F1']['F1']:
#     h_param['F1']['ACC'] = current_ACC
#     h_param['F1']['AUC'] = current_AUC
#     h_param['F1']['F1'] = current_F1
#     h_param['F1']['dilation_radius'] = int(
#         mask.split('_')[1])
#     h_param['F1']['p_center_distance'] = int(
#         mask.split('_')[2])
#     h_param['F1']['Otsu'] = int(mask.split('_')[3])
#     h_param['F1']['n_segments'] = n_segments
#     h_param['F1']['compactness'] = compactness
#     h_param['F1']['p_seeds_final'] = p_seeds_final
#
# if current_ACC > h_param['ACC']['ACC']:
#     h_param['ACC']['ACC'] = current_ACC
#     h_param['ACC']['AUC'] = current_AUC
#     h_param['ACC']['F1'] = current_F1
#     h_param['ACC']['dilation_radius'] = int(
#         mask.split('_')[1])
#     h_param['ACC']['p_center_distance'] = int(
#         mask.split('_')[2])
#     h_param['ACC']['Otsu'] = int(mask.split('_')[3])
#     h_param['ACC']['n_segments'] = n_segments
#     h_param['ACC']['compactness'] = compactness
#     h_param['ACC']['p_seeds_final'] = p_seeds_final