��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-0-ge5bf8de4108��
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@�*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�#
value�"B�" B�"
�
	conv1
	conv2
bn2
	conv3
bn3
flatten
fc
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer-5
layer_with_weights-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�
(axis
	)gamma
*beta
+moving_mean
,moving_variance
-	variables
.trainable_variables
/regularization_losses
0	keras_api
R
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
v
0
1
2
3
4
5
6
7
"8
#9
)10
*11
+12
,13
514
615
V
0
1
2
3
4
5
"6
#7
)8
*9
510
611
 
�
;layer_regularization_losses

<layers
	variables
=metrics
	trainable_variables
>non_trainable_variables

regularization_losses
 
JH
VARIABLE_VALUEconv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
?layer_regularization_losses

@layers
	variables
Ametrics
trainable_variables
Bnon_trainable_variables
regularization_losses
LJ
VARIABLE_VALUEconv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Clayer_regularization_losses

Dlayers
	variables
Emetrics
trainable_variables
Fnon_trainable_variables
regularization_losses
 
US
VARIABLE_VALUEbatch_normalization_2/gamma$bn2/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEbatch_normalization_2/beta#bn2/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE!batch_normalization_2/moving_mean*bn2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE%batch_normalization_2/moving_variance.bn2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�
Glayer_regularization_losses

Hlayers
	variables
Imetrics
trainable_variables
Jnon_trainable_variables
 regularization_losses
LJ
VARIABLE_VALUEconv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
Klayer_regularization_losses

Llayers
$	variables
Mmetrics
%trainable_variables
Nnon_trainable_variables
&regularization_losses
 
US
VARIABLE_VALUEbatch_normalization_3/gamma$bn3/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEbatch_normalization_3/beta#bn3/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE!batch_normalization_3/moving_mean*bn3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE%batch_normalization_3/moving_variance.bn3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
+2
,3

)0
*1
 
�
Olayer_regularization_losses

Players
-	variables
Qmetrics
.trainable_variables
Rnon_trainable_variables
/regularization_losses
 
 
 
�
Slayer_regularization_losses

Tlayers
1	variables
Umetrics
2trainable_variables
Vnon_trainable_variables
3regularization_losses
HF
VARIABLE_VALUEdense_1/kernel$fc/kernel/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEdense_1/bias"fc/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
�
Wlayer_regularization_losses

Xlayers
7	variables
Ymetrics
8trainable_variables
Znon_trainable_variables
9regularization_losses
 
1
0
1
2
3
4
5
6
 

0
1
+2
,3
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 

+0
,1
 
 
 
 
 
 
 
 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_1/kerneldense_1/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_42415098
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_save_42415722
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_1/kerneldense_1/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference__traced_restore_42415782ө

�
�
8__inference_batch_normalization_2_layer_call_fn_42415388

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424147742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
F__inference_conv2d_2_layer_call_and_return_conditional_losses_42414568

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42414857

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_42415643

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�*
�
!__inference__traced_save_42415722
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6c0e82997d97498e9066d3bef42e26db/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn2/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn2/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn3/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn3/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@�:�:�:�:�:�:��:�:�:�:�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_42415628

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�W
�
K__inference_discriminator_layer_call_and_return_conditional_losses_42415260

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d/BiasAddm
	LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:���������@2
	LeakyRelu�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2DLeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_1/BiasAdd�
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_2/LogicalAnd/x�
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/y�
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAnd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_2/Const�
LeakyRelu_1	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
LeakyRelu_1�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2DLeakyRelu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_2/BiasAdd�
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_3/LogicalAnd/x�
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/y�
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAnd�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_3/Const�
LeakyRelu_2	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
LeakyRelu_2o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshapeLeakyRelu_2:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAdd�
IdentityIdentitydense_1/BiasAdd:output:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
F
*__inference_flatten_layer_call_fn_42415633

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_424148882
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415422

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42415407
assignmovingavg_1_42415414
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42415407*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42415407*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42415407*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42415407*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42415407*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42415407AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42415407*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42415414*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415414*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42415414*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415414*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415414*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42415414AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42415414*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
0__inference_discriminator_layer_call_fn_42415281

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_424149822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�E
�	
$__inference__traced_restore_42415782
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias2
.assignvariableop_4_batch_normalization_2_gamma1
-assignvariableop_5_batch_normalization_2_beta8
4assignvariableop_6_batch_normalization_2_moving_mean<
8assignvariableop_7_batch_normalization_2_moving_variance&
"assignvariableop_8_conv2d_2_kernel$
 assignvariableop_9_conv2d_2_bias3
/assignvariableop_10_batch_normalization_3_gamma2
.assignvariableop_11_batch_normalization_3_beta9
5assignvariableop_12_batch_normalization_3_moving_mean=
9assignvariableop_13_batch_normalization_3_moving_variance&
"assignvariableop_14_dense_1_kernel$
 assignvariableop_15_dense_1_bias
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn2/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn2/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn3/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn3/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_2_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_2_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_2_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_2_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_3_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_3_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_3_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_3_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_1_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_1_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16�
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�$
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42414518

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42414503
assignmovingavg_1_42414510
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42414503*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42414503*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42414503*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42414503*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42414503*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42414503AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42414503*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42414510*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414510*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42414510*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414510*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414510*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42414510AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42414510*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42414549

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�/
�	
K__inference_discriminator_layer_call_and_return_conditional_losses_42414919
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_424143962 
conv2d/StatefulPartitionedCall}
	LeakyRelu	LeakyRelu'conv2d/StatefulPartitionedCall:output:0*/
_output_shapes
:���������@2
	LeakyRelu�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_424144162"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424147522/
-batch_normalization_2/StatefulPartitionedCall�
LeakyRelu_1	LeakyRelu6batch_normalization_2/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_1�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_424145682"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424148352/
-batch_normalization_3/StatefulPartitionedCall�
LeakyRelu_2	LeakyRelu6batch_normalization_3/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_2�
flatten/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_424148882
flatten/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_424149062!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_3_layer_call_fn_42415548

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424147012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
F__inference_conv2d_1_layer_call_and_return_conditional_losses_42414416

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
D__inference_conv2d_layer_call_and_return_conditional_losses_42414396

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415530

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_3_layer_call_fn_42415539

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424146702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_42414888

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�/
�	
K__inference_discriminator_layer_call_and_return_conditional_losses_42414949
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_424143962 
conv2d/StatefulPartitionedCall}
	LeakyRelu	LeakyRelu'conv2d/StatefulPartitionedCall:output:0*/
_output_shapes
:���������@2
	LeakyRelu�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_424144162"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424147742/
-batch_normalization_2/StatefulPartitionedCall�
LeakyRelu_1	LeakyRelu6batch_normalization_2/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_1�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_424145682"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424148572/
-batch_normalization_3/StatefulPartitionedCall�
LeakyRelu_2	LeakyRelu6batch_normalization_3/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_2�
flatten/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_424148882
flatten/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_424149062!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
&__inference_signature_wrapper_42415098
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__wrapped_model_424143842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415444

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
+__inference_conv2d_1_layer_call_fn_42414424

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_424144162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�/
�	
K__inference_discriminator_layer_call_and_return_conditional_losses_42415033

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_424143962 
conv2d/StatefulPartitionedCall}
	LeakyRelu	LeakyRelu'conv2d/StatefulPartitionedCall:output:0*/
_output_shapes
:���������@2
	LeakyRelu�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_424144162"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424147742/
-batch_normalization_2/StatefulPartitionedCall�
LeakyRelu_1	LeakyRelu6batch_normalization_2/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_1�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_424145682"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424148572/
-batch_normalization_3/StatefulPartitionedCall�
LeakyRelu_2	LeakyRelu6batch_normalization_3/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_2�
flatten/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_424148882
flatten/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_424149062!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�n
�
#__inference__wrapped_model_42414384
input_17
3discriminator_conv2d_conv2d_readvariableop_resource8
4discriminator_conv2d_biasadd_readvariableop_resource9
5discriminator_conv2d_1_conv2d_readvariableop_resource:
6discriminator_conv2d_1_biasadd_readvariableop_resource?
;discriminator_batch_normalization_2_readvariableop_resourceA
=discriminator_batch_normalization_2_readvariableop_1_resourceP
Ldiscriminator_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceR
Ndiscriminator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource9
5discriminator_conv2d_2_conv2d_readvariableop_resource:
6discriminator_conv2d_2_biasadd_readvariableop_resource?
;discriminator_batch_normalization_3_readvariableop_resourceA
=discriminator_batch_normalization_3_readvariableop_1_resourceP
Ldiscriminator_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceR
Ndiscriminator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4discriminator_dense_1_matmul_readvariableop_resource9
5discriminator_dense_1_biasadd_readvariableop_resource
identity��Cdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�Ediscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�2discriminator/batch_normalization_2/ReadVariableOp�4discriminator/batch_normalization_2/ReadVariableOp_1�Cdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Ediscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�2discriminator/batch_normalization_3/ReadVariableOp�4discriminator/batch_normalization_3/ReadVariableOp_1�+discriminator/conv2d/BiasAdd/ReadVariableOp�*discriminator/conv2d/Conv2D/ReadVariableOp�-discriminator/conv2d_1/BiasAdd/ReadVariableOp�,discriminator/conv2d_1/Conv2D/ReadVariableOp�-discriminator/conv2d_2/BiasAdd/ReadVariableOp�,discriminator/conv2d_2/Conv2D/ReadVariableOp�,discriminator/dense_1/BiasAdd/ReadVariableOp�+discriminator/dense_1/MatMul/ReadVariableOp�
*discriminator/conv2d/Conv2D/ReadVariableOpReadVariableOp3discriminator_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*discriminator/conv2d/Conv2D/ReadVariableOp�
discriminator/conv2d/Conv2DConv2Dinput_12discriminator/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
discriminator/conv2d/Conv2D�
+discriminator/conv2d/BiasAdd/ReadVariableOpReadVariableOp4discriminator_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+discriminator/conv2d/BiasAdd/ReadVariableOp�
discriminator/conv2d/BiasAddBiasAdd$discriminator/conv2d/Conv2D:output:03discriminator/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
discriminator/conv2d/BiasAdd�
discriminator/LeakyRelu	LeakyRelu%discriminator/conv2d/BiasAdd:output:0*/
_output_shapes
:���������@2
discriminator/LeakyRelu�
,discriminator/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5discriminator_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02.
,discriminator/conv2d_1/Conv2D/ReadVariableOp�
discriminator/conv2d_1/Conv2DConv2D%discriminator/LeakyRelu:activations:04discriminator/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
discriminator/conv2d_1/Conv2D�
-discriminator/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6discriminator_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-discriminator/conv2d_1/BiasAdd/ReadVariableOp�
discriminator/conv2d_1/BiasAddBiasAdd&discriminator/conv2d_1/Conv2D:output:05discriminator/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
discriminator/conv2d_1/BiasAdd�
0discriminator/batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 22
0discriminator/batch_normalization_2/LogicalAnd/x�
0discriminator/batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z22
0discriminator/batch_normalization_2/LogicalAnd/y�
.discriminator/batch_normalization_2/LogicalAnd
LogicalAnd9discriminator/batch_normalization_2/LogicalAnd/x:output:09discriminator/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 20
.discriminator/batch_normalization_2/LogicalAnd�
2discriminator/batch_normalization_2/ReadVariableOpReadVariableOp;discriminator_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype024
2discriminator/batch_normalization_2/ReadVariableOp�
4discriminator/batch_normalization_2/ReadVariableOp_1ReadVariableOp=discriminator_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4discriminator/batch_normalization_2/ReadVariableOp_1�
Cdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLdiscriminator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Cdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
Ediscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNdiscriminator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Ediscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
4discriminator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3'discriminator/conv2d_1/BiasAdd:output:0:discriminator/batch_normalization_2/ReadVariableOp:value:0<discriminator/batch_normalization_2/ReadVariableOp_1:value:0Kdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 26
4discriminator/batch_normalization_2/FusedBatchNormV3�
)discriminator/batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2+
)discriminator/batch_normalization_2/Const�
discriminator/LeakyRelu_1	LeakyRelu8discriminator/batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
discriminator/LeakyRelu_1�
,discriminator/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5discriminator_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,discriminator/conv2d_2/Conv2D/ReadVariableOp�
discriminator/conv2d_2/Conv2DConv2D'discriminator/LeakyRelu_1:activations:04discriminator/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
discriminator/conv2d_2/Conv2D�
-discriminator/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6discriminator_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-discriminator/conv2d_2/BiasAdd/ReadVariableOp�
discriminator/conv2d_2/BiasAddBiasAdd&discriminator/conv2d_2/Conv2D:output:05discriminator/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
discriminator/conv2d_2/BiasAdd�
0discriminator/batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 22
0discriminator/batch_normalization_3/LogicalAnd/x�
0discriminator/batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z22
0discriminator/batch_normalization_3/LogicalAnd/y�
.discriminator/batch_normalization_3/LogicalAnd
LogicalAnd9discriminator/batch_normalization_3/LogicalAnd/x:output:09discriminator/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 20
.discriminator/batch_normalization_3/LogicalAnd�
2discriminator/batch_normalization_3/ReadVariableOpReadVariableOp;discriminator_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype024
2discriminator/batch_normalization_3/ReadVariableOp�
4discriminator/batch_normalization_3/ReadVariableOp_1ReadVariableOp=discriminator_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4discriminator/batch_normalization_3/ReadVariableOp_1�
Cdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpLdiscriminator_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Cdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Ediscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNdiscriminator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Ediscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
4discriminator/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3'discriminator/conv2d_2/BiasAdd:output:0:discriminator/batch_normalization_3/ReadVariableOp:value:0<discriminator/batch_normalization_3/ReadVariableOp_1:value:0Kdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Mdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 26
4discriminator/batch_normalization_3/FusedBatchNormV3�
)discriminator/batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2+
)discriminator/batch_normalization_3/Const�
discriminator/LeakyRelu_2	LeakyRelu8discriminator/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
discriminator/LeakyRelu_2�
discriminator/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
discriminator/flatten/Const�
discriminator/flatten/ReshapeReshape'discriminator/LeakyRelu_2:activations:0$discriminator/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
discriminator/flatten/Reshape�
+discriminator/dense_1/MatMul/ReadVariableOpReadVariableOp4discriminator_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+discriminator/dense_1/MatMul/ReadVariableOp�
discriminator/dense_1/MatMulMatMul&discriminator/flatten/Reshape:output:03discriminator/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
discriminator/dense_1/MatMul�
,discriminator/dense_1/BiasAdd/ReadVariableOpReadVariableOp5discriminator_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,discriminator/dense_1/BiasAdd/ReadVariableOp�
discriminator/dense_1/BiasAddBiasAdd&discriminator/dense_1/MatMul:product:04discriminator/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
discriminator/dense_1/BiasAdd�
IdentityIdentity&discriminator/dense_1/BiasAdd:output:0D^discriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^discriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^discriminator/batch_normalization_2/ReadVariableOp5^discriminator/batch_normalization_2/ReadVariableOp_1D^discriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpF^discriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_13^discriminator/batch_normalization_3/ReadVariableOp5^discriminator/batch_normalization_3/ReadVariableOp_1,^discriminator/conv2d/BiasAdd/ReadVariableOp+^discriminator/conv2d/Conv2D/ReadVariableOp.^discriminator/conv2d_1/BiasAdd/ReadVariableOp-^discriminator/conv2d_1/Conv2D/ReadVariableOp.^discriminator/conv2d_2/BiasAdd/ReadVariableOp-^discriminator/conv2d_2/Conv2D/ReadVariableOp-^discriminator/dense_1/BiasAdd/ReadVariableOp,^discriminator/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2�
Cdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCdiscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
Ediscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ediscriminator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2discriminator/batch_normalization_2/ReadVariableOp2discriminator/batch_normalization_2/ReadVariableOp2l
4discriminator/batch_normalization_2/ReadVariableOp_14discriminator/batch_normalization_2/ReadVariableOp_12�
Cdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpCdiscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Ediscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ediscriminator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12h
2discriminator/batch_normalization_3/ReadVariableOp2discriminator/batch_normalization_3/ReadVariableOp2l
4discriminator/batch_normalization_3/ReadVariableOp_14discriminator/batch_normalization_3/ReadVariableOp_12Z
+discriminator/conv2d/BiasAdd/ReadVariableOp+discriminator/conv2d/BiasAdd/ReadVariableOp2X
*discriminator/conv2d/Conv2D/ReadVariableOp*discriminator/conv2d/Conv2D/ReadVariableOp2^
-discriminator/conv2d_1/BiasAdd/ReadVariableOp-discriminator/conv2d_1/BiasAdd/ReadVariableOp2\
,discriminator/conv2d_1/Conv2D/ReadVariableOp,discriminator/conv2d_1/Conv2D/ReadVariableOp2^
-discriminator/conv2d_2/BiasAdd/ReadVariableOp-discriminator/conv2d_2/BiasAdd/ReadVariableOp2\
,discriminator/conv2d_2/Conv2D/ReadVariableOp,discriminator/conv2d_2/Conv2D/ReadVariableOp2\
,discriminator/dense_1/BiasAdd/ReadVariableOp,discriminator/dense_1/BiasAdd/ReadVariableOp2Z
+discriminator/dense_1/MatMul/ReadVariableOp+discriminator/dense_1/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
�
�
)__inference_conv2d_layer_call_fn_42414404

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_424143962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_discriminator_layer_call_fn_42415302

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_424150332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_discriminator_layer_call_fn_42415052
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_424150332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_2_layer_call_fn_42415379

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424147522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42414670

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42414655
assignmovingavg_1_42414662
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42414655*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42414655*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42414655*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42414655*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42414655*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42414655AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42414655*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42414662*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414662*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42414662*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414662*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414662*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42414662AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42414662*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415604

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42414835

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42414820
assignmovingavg_1_42414827
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42414820*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42414820*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42414820*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42414820*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42414820*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42414820AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42414820*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42414827*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414827*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42414827*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414827*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414827*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42414827AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42414827*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_42414906

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_3_layer_call_fn_42415613

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424148352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_3_layer_call_fn_42415622

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424148572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415508

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42415493
assignmovingavg_1_42415500
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42415493*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42415493*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42415493*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42415493*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42415493*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42415493AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42415493*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42415500*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415500*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42415500*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415500*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415500*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42415500AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42415500*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
+__inference_conv2d_2_layer_call_fn_42414576

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_424145682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42414752

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42414737
assignmovingavg_1_42414744
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42414737*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42414737*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42414737*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42414737*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42414737*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42414737AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42414737*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42414744*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414744*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42414744*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414744*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42414744*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42414744AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42414744*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_2_layer_call_fn_42415453

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424145182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
��
�
K__inference_discriminator_layer_call_and_return_conditional_losses_42415191

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource2
.batch_normalization_2_assignmovingavg_424151304
0batch_normalization_2_assignmovingavg_1_42415137+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resource2
.batch_normalization_3_assignmovingavg_424151674
0batch_normalization_3_assignmovingavg_1_42415174*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d/BiasAddm
	LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:���������@2
	LeakyRelu�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2DLeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_1/BiasAdd�
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/x�
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/y�
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAnd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_2/ReadVariableOp_1}
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const�
batch_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2(
&batch_normalization_2/FusedBatchNormV3�
batch_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_2/Const_2�
+batch_normalization_2/AssignMovingAvg/sub/xConst*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/42415130*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_2/AssignMovingAvg/sub/x�
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/42415130*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/sub�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_42415130*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/42415130*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg/sub_1�
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/42415130*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/mul�
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_42415130-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/42415130*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/42415137*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x�
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/42415137*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/sub�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_2_assignmovingavg_1_42415137*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/42415137*
_output_shapes	
:�2/
-batch_normalization_2/AssignMovingAvg_1/sub_1�
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/42415137*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/mul�
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_2_assignmovingavg_1_42415137/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/42415137*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�
LeakyRelu_1	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
LeakyRelu_1�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2DLeakyRelu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_2/BiasAdd�
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/x�
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/y�
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAnd�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_3/ReadVariableOp_1}
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_3/Const�
batch_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_3/Const_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0$batch_normalization_3/Const:output:0&batch_normalization_3/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2(
&batch_normalization_3/FusedBatchNormV3�
batch_normalization_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_3/Const_2�
+batch_normalization_3/AssignMovingAvg/sub/xConst*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/42415167*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_3/AssignMovingAvg/sub/x�
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/42415167*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/sub�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_42415167*
_output_shapes	
:�*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/42415167*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg/sub_1�
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/42415167*
_output_shapes	
:�2+
)batch_normalization_3/AssignMovingAvg/mul�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_42415167-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/42415167*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/42415174*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_3/AssignMovingAvg_1/sub/x�
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/42415174*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/sub�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_3_assignmovingavg_1_42415174*
_output_shapes	
:�*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/42415174*
_output_shapes	
:�2/
-batch_normalization_3/AssignMovingAvg_1/sub_1�
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/42415174*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg_1/mul�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_3_assignmovingavg_1_42415174/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/42415174*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
LeakyRelu_2	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
LeakyRelu_2o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshapeLeakyRelu_2:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAdd�
IdentityIdentitydense_1/BiasAdd:output:0:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_2_layer_call_fn_42415462

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424145492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415370

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�/
�	
K__inference_discriminator_layer_call_and_return_conditional_losses_42414982

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_424143962 
conv2d/StatefulPartitionedCall}
	LeakyRelu	LeakyRelu'conv2d/StatefulPartitionedCall:output:0*/
_output_shapes
:���������@2
	LeakyRelu�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_424144162"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_424147522/
-batch_normalization_2/StatefulPartitionedCall�
LeakyRelu_1	LeakyRelu6batch_normalization_2/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_1�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_424145682"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_424148352/
-batch_normalization_3/StatefulPartitionedCall�
LeakyRelu_2	LeakyRelu6batch_normalization_3/StatefulPartitionedCall:output:0*0
_output_shapes
:����������2
LeakyRelu_2�
flatten/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_424148882
flatten/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_424149062!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42414701

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
0__inference_discriminator_layer_call_fn_42415001
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_424149822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������@@::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42414774

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415582

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42415567
assignmovingavg_1_42415574
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42415567*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42415567*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42415567*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42415567*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42415567*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42415567AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42415567*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42415574*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415574*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42415574*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415574*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415574*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42415574AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42415574*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415348

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_42415333
assignmovingavg_1_42415340
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/42415333*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/42415333*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42415333*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/42415333*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/42415333*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42415333AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/42415333*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/42415340*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415340*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42415340*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415340*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/42415340*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42415340AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/42415340*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
*__inference_dense_1_layer_call_fn_42415650

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_424149062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@@<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	conv1
	conv2
bn2
	conv3
bn3
flatten
fc
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer-5
layer_with_weights-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
[__call__
*\&call_and_return_all_conditional_losses
]_default_save_signature"�
_tf_keras_model�{"class_name": "Discriminator", "name": "discriminator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Discriminator"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*^&call_and_return_all_conditional_losses
___call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
*b&call_and_return_all_conditional_losses
c__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
*d&call_and_return_all_conditional_losses
e__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [5, 5], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
�
(axis
	)gamma
*beta
+moving_mean
,moving_variance
-	variables
.trainable_variables
/regularization_losses
0	keras_api
*f&call_and_return_all_conditional_losses
g__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}}
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
*h&call_and_return_all_conditional_losses
i__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
*j&call_and_return_all_conditional_losses
k__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
�
0
1
2
3
4
5
6
7
"8
#9
)10
*11
+12
,13
514
615"
trackable_list_wrapper
v
0
1
2
3
4
5
"6
#7
)8
*9
510
611"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;layer_regularization_losses

<layers
	variables
=metrics
	trainable_variables
>non_trainable_variables

regularization_losses
[__call__
]_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
lserving_default"
signature_map
':%@2conv2d/kernel
:@2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?layer_regularization_losses

@layers
	variables
Ametrics
trainable_variables
Bnon_trainable_variables
regularization_losses
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
*:(@�2conv2d_1/kernel
:�2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Clayer_regularization_losses

Dlayers
	variables
Emetrics
trainable_variables
Fnon_trainable_variables
regularization_losses
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Glayer_regularization_losses

Hlayers
	variables
Imetrics
trainable_variables
Jnon_trainable_variables
 regularization_losses
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
+:)��2conv2d_2/kernel
:�2conv2d_2/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Klayer_regularization_losses

Llayers
$	variables
Mmetrics
%trainable_variables
Nnon_trainable_variables
&regularization_losses
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_3/gamma
):'�2batch_normalization_3/beta
2:0� (2!batch_normalization_3/moving_mean
6:4� (2%batch_normalization_3/moving_variance
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Olayer_regularization_losses

Players
-	variables
Qmetrics
.trainable_variables
Rnon_trainable_variables
/regularization_losses
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Slayer_regularization_losses

Tlayers
1	variables
Umetrics
2trainable_variables
Vnon_trainable_variables
3regularization_losses
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_1/kernel
:2dense_1/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wlayer_regularization_losses

Xlayers
7	variables
Ymetrics
8trainable_variables
Znon_trainable_variables
9regularization_losses
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
+2
,3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
0__inference_discriminator_layer_call_fn_42415001
0__inference_discriminator_layer_call_fn_42415052
0__inference_discriminator_layer_call_fn_42415302
0__inference_discriminator_layer_call_fn_42415281�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_discriminator_layer_call_and_return_conditional_losses_42414949
K__inference_discriminator_layer_call_and_return_conditional_losses_42415191
K__inference_discriminator_layer_call_and_return_conditional_losses_42415260
K__inference_discriminator_layer_call_and_return_conditional_losses_42414919�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_42414384�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������@@
�2�
D__inference_conv2d_layer_call_and_return_conditional_losses_42414396�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
)__inference_conv2d_layer_call_fn_42414404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
F__inference_conv2d_1_layer_call_and_return_conditional_losses_42414416�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
+__inference_conv2d_1_layer_call_fn_42414424�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415444
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415422
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415370
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415348�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_2_layer_call_fn_42415453
8__inference_batch_normalization_2_layer_call_fn_42415462
8__inference_batch_normalization_2_layer_call_fn_42415379
8__inference_batch_normalization_2_layer_call_fn_42415388�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_conv2d_2_layer_call_and_return_conditional_losses_42414568�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
+__inference_conv2d_2_layer_call_fn_42414576�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415508
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415582
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415604
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415530�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_3_layer_call_fn_42415548
8__inference_batch_normalization_3_layer_call_fn_42415539
8__inference_batch_normalization_3_layer_call_fn_42415622
8__inference_batch_normalization_3_layer_call_fn_42415613�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_flatten_layer_call_and_return_conditional_losses_42415628�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_layer_call_fn_42415633�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_1_layer_call_and_return_conditional_losses_42415643�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_1_layer_call_fn_42415650�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5B3
&__inference_signature_wrapper_42415098input_1�
#__inference__wrapped_model_42414384�"#)*+,568�5
.�+
)�&
input_1���������@@
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415348t<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415370t<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415422�N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42415444�N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
8__inference_batch_normalization_2_layer_call_fn_42415379g<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_2_layer_call_fn_42415388g<�9
2�/
)�&
inputs����������
p 
� "!������������
8__inference_batch_normalization_2_layer_call_fn_42415453�N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_2_layer_call_fn_42415462�N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415508�)*+,N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415530�)*+,N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415582t)*+,<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_42415604t)*+,<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
8__inference_batch_normalization_3_layer_call_fn_42415539�)*+,N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_3_layer_call_fn_42415548�)*+,N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_3_layer_call_fn_42415613g)*+,<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_3_layer_call_fn_42415622g)*+,<�9
2�/
)�&
inputs����������
p 
� "!������������
F__inference_conv2d_1_layer_call_and_return_conditional_losses_42414416�I�F
?�<
:�7
inputs+���������������������������@
� "@�=
6�3
0,����������������������������
� �
+__inference_conv2d_1_layer_call_fn_42414424�I�F
?�<
:�7
inputs+���������������������������@
� "3�0,�����������������������������
F__inference_conv2d_2_layer_call_and_return_conditional_losses_42414568�"#J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
+__inference_conv2d_2_layer_call_fn_42414576�"#J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
D__inference_conv2d_layer_call_and_return_conditional_losses_42414396�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������@
� �
)__inference_conv2d_layer_call_fn_42414404�I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������@�
E__inference_dense_1_layer_call_and_return_conditional_losses_42415643]560�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_1_layer_call_fn_42415650P560�-
&�#
!�
inputs����������
� "�����������
K__inference_discriminator_layer_call_and_return_conditional_losses_42414919w"#)*+,56<�9
2�/
)�&
input_1���������@@
p
� "%�"
�
0���������
� �
K__inference_discriminator_layer_call_and_return_conditional_losses_42414949w"#)*+,56<�9
2�/
)�&
input_1���������@@
p 
� "%�"
�
0���������
� �
K__inference_discriminator_layer_call_and_return_conditional_losses_42415191v"#)*+,56;�8
1�.
(�%
inputs���������@@
p
� "%�"
�
0���������
� �
K__inference_discriminator_layer_call_and_return_conditional_losses_42415260v"#)*+,56;�8
1�.
(�%
inputs���������@@
p 
� "%�"
�
0���������
� �
0__inference_discriminator_layer_call_fn_42415001j"#)*+,56<�9
2�/
)�&
input_1���������@@
p
� "�����������
0__inference_discriminator_layer_call_fn_42415052j"#)*+,56<�9
2�/
)�&
input_1���������@@
p 
� "�����������
0__inference_discriminator_layer_call_fn_42415281i"#)*+,56;�8
1�.
(�%
inputs���������@@
p
� "�����������
0__inference_discriminator_layer_call_fn_42415302i"#)*+,56;�8
1�.
(�%
inputs���������@@
p 
� "�����������
E__inference_flatten_layer_call_and_return_conditional_losses_42415628b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
*__inference_flatten_layer_call_fn_42415633U8�5
.�+
)�&
inputs����������
� "������������
&__inference_signature_wrapper_42415098�"#)*+,56C�@
� 
9�6
4
input_1)�&
input_1���������@@"3�0
.
output_1"�
output_1���������