import device.exp_device
import device.opportunistic_device
import device.hetero_device
import device.tmc_exp_device
import device.federated_device
import device.quantization_device
import device.droppcl_device
import device.gr_device
import device.gossip_device

def get_device_class(class_name):
    ### Hetero & Dropout
    if class_name == 'hetero':
        return device.hetero_device.HeteroDevice
    elif class_name == 'dropin':
        return device.hetero_device.DropinDevice
    elif class_name == 'dropinnout':
        return device.hetero_device.DropInNOutDevice
    elif class_name == 'dropout only':
        return device.hetero_device.DropoutOnlyOnDevice
    elif class_name == 'mixed dropout':
        return device.hetero_device.MixedDropoutDevice
    elif class_name == 'mixed dropout 2':
        return device.hetero_device.MixedDropoutDevice
    elif class_name == 'dyn mixed dropout':
        return device.hetero_device.DynamicMixedDropoutDevice
    elif class_name == 'no dropout':
        return device.hetero_device.NoDropoutDevice
    elif class_name == 'mixed scaled dropout':
        return device.hetero_device.MixedScaledDropoutDevice
    elif class_name == 'mixed multiopt dropout':
        return device.hetero_device.MixedMultiOptDropoutDevice 
    elif class_name == 'momentum dropout':
        return device.hetero_device.MomentumMixedDropoutDevice
    elif class_name == 'auto m. dropout':
        return device.hetero_device.AutoMomentumMixedDropoutDevice

    ### Quantization
    elif class_name == 'Q. grad':
        return device.quantization_device.QuantizationDevice
    elif class_name == 'No Q':
        return device.quantization_device.NoQuantizationDevice
    elif class_name == 'Q. params':
        return device.quantization_device.QuantizationParamDevice
    elif class_name == 'Q. grad & params':
        return device.quantization_device.QuantizationGradParamDevice
    elif class_name == 'mixed Q. grad':
        return device.quantization_device.MixedQuantizationDevice
    elif class_name == 'Q. Net':
        return device.quantization_device.QuantizationNetworksDevice

    ### DROppCL Devices for final exp.
    elif class_name == 'DROppCL test':
        return device.droppcl_device.DROppCLTestDevice
    elif class_name == 'baseline':
        return device.droppcl_device.DROppCLBaselineDevice
    elif class_name == 'dropout':
        return device.droppcl_device.DROppCLOnlyDropoutDevice
    elif class_name == 'quantize':
        return device.droppcl_device.DROppCLOnlyQuantizationDevice
    elif class_name == 'DROppCL':
        return device.droppcl_device.DROppCLDevice

    ### DROppCL Devices for controlled exp.
    elif class_name == 'c_DROppCL':
        return device.quantization_device.MomentumDROppCLDevice
    elif class_name == 'c_DROppCL Auto':
        return device.quantization_device.AutoMomentumDROppCLDevice
    elif class_name == 'c_only dropout':
        return device.quantization_device.OnlyDropoutDevice
    elif class_name == 'c_only quant':
        return device.quantization_device.OnlyQuantDevice
    elif class_name == 'no dropout nor Q.':
        return device.quantization_device.NoDropoutNorQDevice
        
    #####################
    if class_name == 'greedy':
        return device.exp_device.GreedyWOSimDevice
    elif class_name == 'local':
        return device.exp_device.LocalDevice
    elif class_name == 'opportunistic':
        return device.opportunistic_device.JSDOppDevice
    elif class_name == 'opportunistic-weighted':
        return device.tmc_exp_device.JSDOppWeightedDevice
    elif class_name == 'opportunistic (low thres.)':
        return device.opportunistic_device.LowJSDOppDevice
    elif class_name == 'federated':
        return device.federated_device.FederatedDevice
    elif class_name == 'federated (opportunistic)':
        return device.federated_device.FederatedJSDGreedyDevice
    elif class_name == 'gradient replay':
        return device.gr_device.JSDGradientReplayDevice

    ##### OVM FL Devices
    if class_name == 'fl-server':
        return device.federated_device.OVMFLServerDevice
    elif class_name == 'fl-client':
        return device.federated_device.OVMFLClientDevice

    ##### Other OVM Devices
    if class_name == 'gossip':
        return device.gossip_device.OVMGossipDevice
    elif class_name == 'b-gossip':
        return device.gossip_device.OVMBroadcastGossipDevice

    raise ValueError('Cannot find device name {}'.format(class_name))
