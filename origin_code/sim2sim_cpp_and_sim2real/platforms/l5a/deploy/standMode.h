/*
 * File: standMode.h
 *
 * Code generated for Simulink model 'standMode'.
 *
 * Model version                  : 1.578
 * Simulink Coder version         : 9.8 (R2022b) 13-May-2022
 * C/C++ source code generated on : Tue Dec 17 10:31:17 2024
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Execution efficiency
 * Validation result: Not run
 */
#pragma once

#ifndef RTW_HEADER_standMode_h_
#  define RTW_HEADER_standMode_h_
#  ifndef standMode_COMMON_INCLUDES_
#    define standMode_COMMON_INCLUDES_
#    include <stdio.h>
#    include "rtwtypes.h"
#  endif /* standMode_COMMON_INCLUDES_ */

#  include "standMode_types.h"
// #include "rt_nonfinite.h"
// #include "rtGetInf.h"

/* Macros for accessing real-time model data structure */
#  ifndef rtmGetErrorStatus
#    define rtmGetErrorStatus(rtm) ((rtm)->errorStatus)
#  endif

#  ifndef rtmSetErrorStatus
#    define rtmSetErrorStatus(rtm, val) ((rtm)->errorStatus = (val))
#  endif

/* Block signals and states (default storage) for system '<Root>' */
typedef struct {
  sws399s4figUQr0kdVXer9B_stand_T state;      /* '<S1>/MATLAB Function13' */
  s9jYAhnLwYvBxffH4z6wNKG_stand_T state_f;    /* '<S1>/MATLAB Function12' */
  s9jYAhnLwYvBxffH4z6wNKG_stand_T state_e;    /* '<S1>/MATLAB Function18' */
  s9jYAhnLwYvBxffH4z6wNKG_stand_T state_k;    /* '<S1>/MATLAB Function19' */
  s9jYAhnLwYvBxffH4z6wNKG_stand_T state_a;    /* '<S1>/MATLAB Function20' */
  stRJlYAl3pjTyFPnByxQ8b_standM_T Objstate;   /* '<S1>/MATLAB Function1' */
  stRJlYAl3pjTyFPnByxQ8b_standM_T Objstate_c; /* '<S1>/MATLAB Function2' */
  real_T OutportBufferForswitch0[10];
  real_T pos_l[6];                    /* '<S91>/MATLAB Function' */
  real_T pos_r[6];                    /* '<S91>/MATLAB Function' */
  real_T mode;                        /* '<S3>/Chart' */
  real_T V_des_out;                   /* '<S3>/Chart' */
  real_T W_des_out;                   /* '<S3>/Chart' */
  real_T operation_mode;              /* '<S3>/Chart' */
  real_T PID_reset_posCtrl;           /* '<S3>/Chart' */
  real_T Integrator_DSTATE;           /* '<S148>/Integrator' */
  real_T Filter_DSTATE;               /* '<S143>/Filter' */
  real_T Integrator_DSTATE_f;         /* '<S197>/Integrator' */
  real_T Filter_DSTATE_f;             /* '<S192>/Filter' */
  real_T PrevY;                       /* '<S1>/Rate Limiter' */
  real_T Memory_PreviousInput;        /* '<S1>/Memory' */
  real_T Memory1_PreviousInput;       /* '<S1>/Memory1' */
  real_T time;                        /* '<S91>/timer' */
  real_T num;                         /* '<S91>/MATLAB Function' */
  real_T zl;                          /* '<S91>/MATLAB Function' */
  real_T zr;                          /* '<S91>/MATLAB Function' */
  real_T once;                        /* '<S3>/Chart' */
  real_T errorState_count;            /* '<S3>/Chart' */
  real_T PID_reset_count;             /* '<S3>/Chart' */
  real_T IMU_comp_count;              /* '<S1>/MATLAB Function9' */
  real_T StepKal;                     /* '<S1>/MATLAB Function2' */
  real_T StepKal_p;                   /* '<S1>/MATLAB Function1' */
  int8_T Integrator_PrevResetState;   /* '<S148>/Integrator' */
  int8_T Filter_PrevResetState;       /* '<S143>/Filter' */
  int8_T Integrator_PrevResetState_l; /* '<S197>/Integrator' */
  int8_T Filter_PrevResetState_k;     /* '<S192>/Filter' */
  uint8_T is_c8_standMode;            /* '<S3>/Chart' */
  uint8_T is_active_c8_standMode;     /* '<S3>/Chart' */
  boolean_T isIni;                    /* '<S91>/MATLAB Function' */
  boolean_T isIni_not_empty;          /* '<S91>/MATLAB Function' */
  boolean_T state_not_empty;          /* '<S1>/MATLAB Function20' */
  boolean_T state_not_empty_f;        /* '<S1>/MATLAB Function19' */
  boolean_T state_not_empty_p;        /* '<S1>/MATLAB Function18' */
  boolean_T state_not_empty_e;        /* '<S1>/MATLAB Function13' */
  boolean_T state_not_empty_j;        /* '<S1>/MATLAB Function12' */
} DW_standMode_T;

/* Real-time Model Data Structure */
struct tag_RTM_standMode_T {
  const char_T* volatile errorStatus;
};

/* Block signals and states (default storage) */
extern DW_standMode_T standMode_DW;

/* Model entry point functions */
// extern void standMode_initialize(void);
extern void standMode_initialize(standmode_output_t* standmode_output,
                                 standmode_input_t* standmode_inputvoid);
extern void standMode_terminate(void);

/* Customized model step function */
extern void standMode_step(standmode_output_t* standmode_output,
                           standmode_input_t* standmode_input);

void setActuatorParams(standmode_output_t* standmode_output);

void recordData(standmode_output_t* standmode_output,
                standmode_input_t* standmode_input, y4a::RobotModel& y4a_robot);

void setUserCmd(standmode_input_t* standmode_input);

/* Real-time Model object */
extern RT_MODEL_standMode_T* const standMode_M;

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<Root>/Constant1' : Unused code path elimination
 * Block '<Root>/Constant2' : Unused code path elimination
 * Block '<S36>/Compare' : Unused code path elimination
 * Block '<S36>/Constant' : Unused code path elimination
 * Block '<S38>/Compare' : Unused code path elimination
 * Block '<S38>/Constant' : Unused code path elimination
 * Block '<S1>/Data Type Conversion1' : Unused code path elimination
 * Block '<S3>/Constant4' : Unused code path elimination
 * Block '<S102>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S103>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S104>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S105>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S106>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S107>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S108>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S109>/FixPt Data Type Duplicate' : Unused code path elimination
 * Block '<S90>/Data Type Conversion' : Eliminate redundant data type conversion
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'standMode'
 * '<S1>'   : 'standMode/Subsystem'
 * '<S2>'   : 'standMode/Subsystem2'
 * '<S3>'   : 'standMode/controller2'
 * '<S4>'   : 'standMode/Subsystem/Angle Conversion1'
 * '<S5>'   : 'standMode/Subsystem/Angle Conversion11'
 * '<S6>'   : 'standMode/Subsystem/Angle Conversion12'
 * '<S7>'   : 'standMode/Subsystem/Angle Conversion13'
 * '<S8>'   : 'standMode/Subsystem/Angle Conversion14'
 * '<S9>'   : 'standMode/Subsystem/Angle Conversion15'
 * '<S10>'  : 'standMode/Subsystem/Angle Conversion16'
 * '<S11>'  : 'standMode/Subsystem/Angle Conversion17'
 * '<S12>'  : 'standMode/Subsystem/Angle Conversion18'
 * '<S13>'  : 'standMode/Subsystem/Angle Conversion19'
 * '<S14>'  : 'standMode/Subsystem/Angle Conversion2'
 * '<S15>'  : 'standMode/Subsystem/Angle Conversion20'
 * '<S16>'  : 'standMode/Subsystem/Angle Conversion21'
 * '<S17>'  : 'standMode/Subsystem/Angle Conversion22'
 * '<S18>'  : 'standMode/Subsystem/Angle Conversion23'
 * '<S19>'  : 'standMode/Subsystem/Angle Conversion24'
 * '<S20>'  : 'standMode/Subsystem/Angle Conversion3'
 * '<S21>'  : 'standMode/Subsystem/Angle Conversion4'
 * '<S22>'  : 'standMode/Subsystem/Angle Conversion5'
 * '<S23>'  : 'standMode/Subsystem/Angle Conversion6'
 * '<S24>'  : 'standMode/Subsystem/Angle Conversion8'
 * '<S25>'  : 'standMode/Subsystem/Angle Conversion9'
 * '<S26>'  : 'standMode/Subsystem/Angular Velocity Conversion'
 * '<S27>'  : 'standMode/Subsystem/Angular Velocity Conversion1'
 * '<S28>'  : 'standMode/Subsystem/Angular Velocity Conversion2'
 * '<S29>'  : 'standMode/Subsystem/Angular Velocity Conversion3'
 * '<S30>'  : 'standMode/Subsystem/Angular Velocity Conversion4'
 * '<S31>'  : 'standMode/Subsystem/Angular Velocity Conversion5'
 * '<S32>'  : 'standMode/Subsystem/Angular Velocity Conversion6'
 * '<S33>'  : 'standMode/Subsystem/Angular Velocity Conversion7'
 * '<S34>'  : 'standMode/Subsystem/Angular Velocity Conversion8'
 * '<S35>'  : 'standMode/Subsystem/Angular Velocity Conversion9'
 * '<S36>'  : 'standMode/Subsystem/Compare To Constant1'
 * '<S37>'  : 'standMode/Subsystem/Compare To Constant4'
 * '<S38>'  : 'standMode/Subsystem/Compare To Constant5'
 * '<S39>'  : 'standMode/Subsystem/Compare To Constant6'
 * '<S40>'  : 'standMode/Subsystem/Compare To Constant7'
 * '<S41>'  : 'standMode/Subsystem/IMUDataPreprocessing'
 * '<S42>'  : 'standMode/Subsystem/MATLAB Function1'
 * '<S43>'  : 'standMode/Subsystem/MATLAB Function10'
 * '<S44>'  : 'standMode/Subsystem/MATLAB Function11'
 * '<S45>'  : 'standMode/Subsystem/MATLAB Function12'
 * '<S46>'  : 'standMode/Subsystem/MATLAB Function13'
 * '<S47>'  : 'standMode/Subsystem/MATLAB Function18'
 * '<S48>'  : 'standMode/Subsystem/MATLAB Function19'
 * '<S49>'  : 'standMode/Subsystem/MATLAB Function2'
 * '<S50>'  : 'standMode/Subsystem/MATLAB Function20'
 * '<S51>'  : 'standMode/Subsystem/MATLAB Function7'
 * '<S52>'  : 'standMode/Subsystem/MATLAB Function9'
 * '<S53>'  : 'standMode/Subsystem/Velocity Conversion'
 * '<S54>'  : 'standMode/Subsystem2/Arm_interfaces'
 * '<S55>'  : 'standMode/Subsystem2/Radians to Degrees'
 * '<S56>'  : 'standMode/Subsystem2/Radians to Degrees1'
 * '<S57>'  : 'standMode/Subsystem2/Radians to Degrees2'
 * '<S58>'  : 'standMode/Subsystem2/Radians to Degrees3'
 * '<S59>'  : 'standMode/Subsystem2/Radians to Degrees4'
 * '<S60>'  : 'standMode/Subsystem2/Radians to Degrees5'
 * '<S61>'  : 'standMode/Subsystem2/Radians to Degrees6'
 * '<S62>'  : 'standMode/Subsystem2/Radians to Degrees7'
 * '<S63>'  : 'standMode/Subsystem2/Wheel_leged interfaces'
 * '<S64>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem1'
 * '<S65>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem10'
 * '<S66>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem11'
 * '<S67>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem12'
 * '<S68>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem13'
 * '<S69>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem14'
 * '<S70>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem2'
 * '<S71>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem3'
 * '<S72>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem4'
 * '<S73>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem5'
 * '<S74>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem6'
 * '<S75>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem7'
 * '<S76>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem8'
 * '<S77>'  : 'standMode/Subsystem2/Arm_interfaces/Subsystem9'
 * '<S78>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Subsystem1'
 * '<S79>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Subsystem2'
 * '<S80>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Subsystem5'
 * '<S81>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Subsystem6'
 * '<S82>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Subsystem7'
 * '<S83>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Subsystem8'
 * '<S84>'  : 'standMode/Subsystem2/Wheel_leged interfaces/Wheel_L'
 * '<S85>'  : 'standMode/Subsystem2/Wheel_leged interfaces/hip_pitch_L'
 * '<S86>'  : 'standMode/Subsystem2/Wheel_leged interfaces/hip_pitch_R'
 * '<S87>'  : 'standMode/Subsystem2/Wheel_leged interfaces/wheel_R'
 * '<S88>'  : 'standMode/controller2/Chart'
 * '<S89>'  : 'standMode/controller2/Subsystem'
 * '<S90>'  : 'standMode/controller2/Subsystem1'
 * '<S91>'  : 'standMode/controller2/initial_state'
 * '<S92>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion'
 * '<S93>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion1'
 * '<S94>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion2'
 * '<S95>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion3'
 * '<S96>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion4'
 * '<S97>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion5'
 * '<S98>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion6'
 * '<S99>'  : 'standMode/controller2/Subsystem1/Angular Velocity Conversion7'
 * '<S100>' : 'standMode/controller2/Subsystem1/Angular Velocity Conversion8'
 * '<S101>' : 'standMode/controller2/Subsystem1/Angular Velocity Conversion9'
 * '<S102>' : 'standMode/controller2/Subsystem1/Interval Test'
 * '<S103>' : 'standMode/controller2/Subsystem1/Interval Test1'
 * '<S104>' : 'standMode/controller2/Subsystem1/Interval Test2'
 * '<S105>' : 'standMode/controller2/Subsystem1/Interval Test3'
 * '<S106>' : 'standMode/controller2/Subsystem1/Interval Test4'
 * '<S107>' : 'standMode/controller2/Subsystem1/Interval Test5'
 * '<S108>' : 'standMode/controller2/Subsystem1/Interval Test6'
 * '<S109>' : 'standMode/controller2/Subsystem1/Interval Test7'
 * '<S110>' : 'standMode/controller2/initial_state/Drive&Steering torCtl'
 * '<S111>' : 'standMode/controller2/initial_state/IK1'
 * '<S112>' : 'standMode/controller2/initial_state/MATLAB Function'
 * '<S113>' : 'standMode/controller2/initial_state/timer'
 * '<S114>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering'
 * '<S115>' : 'standMode/controller2/initial_state/Drive&Steering torCtl/drive'
 * '<S116>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1'
 * '<S117>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Anti-windup'
 * '<S118>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/D Gain'
 * '<S119>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Filter'
 * '<S120>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Filter ICs'
 * '<S121>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/I Gain'
 * '<S122>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Ideal P Gain'
 * '<S123>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Ideal P Gain Fdbk'
 * '<S124>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Integrator'
 * '<S125>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Integrator ICs'
 * '<S126>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/N Copy'
 * '<S127>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/N Gain'
 * '<S128>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/P Copy'
 * '<S129>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Parallel P Gain'
 * '<S130>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Reset Signal'
 * '<S131>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Saturation'
 * '<S132>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Saturation Fdbk'
 * '<S133>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Sum'
 * '<S134>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Sum Fdbk'
 * '<S135>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tracking Mode'
 * '<S136>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tracking Mode Sum'
 * '<S137>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tsamp - Integral'
 * '<S138>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tsamp - Ngain'
 * '<S139>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/postSat Signal'
 * '<S140>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/preSat Signal'
 * '<S141>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Anti-windup/Passthrough'
 * '<S142>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/D Gain/External Parameters'
 * '<S143>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Filter/Disc. Forward Euler Filter'
 * '<S144>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Filter ICs/Internal IC - Filter'
 * '<S145>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/I Gain/External Parameters'
 * '<S146>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Ideal P Gain/Passthrough'
 * '<S147>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Ideal P Gain Fdbk/Disabled'
 * '<S148>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Integrator/Discrete'
 * '<S149>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Integrator ICs/Internal IC'
 * '<S150>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/N Copy/Disabled'
 * '<S151>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/N Gain/External Parameters'
 * '<S152>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/P Copy/Disabled'
 * '<S153>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Parallel P Gain/External Parameters'
 * '<S154>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Reset Signal/External Reset'
 * '<S155>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Saturation/Passthrough'
 * '<S156>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Saturation Fdbk/Disabled'
 * '<S157>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Sum/Sum_PID'
 * '<S158>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Sum Fdbk/Disabled'
 * '<S159>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tracking Mode/Disabled'
 * '<S160>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tracking Mode Sum/Passthrough'
 * '<S161>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tsamp - Integral/Passthrough'
 * '<S162>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/Tsamp - Ngain/Passthrough'
 * '<S163>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/postSat Signal/Forward_Path'
 * '<S164>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/Steering/Discrete PID Controller1/preSat Signal/Forward_Path'
 * '<S165>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1'
 * '<S166>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Anti-windup'
 * '<S167>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/D Gain'
 * '<S168>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Filter'
 * '<S169>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Filter ICs'
 * '<S170>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/I Gain'
 * '<S171>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Ideal P Gain'
 * '<S172>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Ideal P Gain Fdbk'
 * '<S173>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Integrator'
 * '<S174>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Integrator ICs'
 * '<S175>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/N Copy'
 * '<S176>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/N Gain'
 * '<S177>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/P Copy'
 * '<S178>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Parallel P Gain'
 * '<S179>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Reset Signal'
 * '<S180>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Saturation'
 * '<S181>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Saturation Fdbk'
 * '<S182>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Sum'
 * '<S183>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Sum Fdbk'
 * '<S184>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tracking Mode'
 * '<S185>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tracking Mode Sum'
 * '<S186>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tsamp - Integral'
 * '<S187>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tsamp - Ngain'
 * '<S188>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/postSat Signal'
 * '<S189>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/preSat Signal'
 * '<S190>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Anti-windup/Passthrough'
 * '<S191>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/D Gain/External Parameters'
 * '<S192>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Filter/Disc. Forward Euler Filter'
 * '<S193>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Filter ICs/Internal IC - Filter'
 * '<S194>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/I Gain/External Parameters'
 * '<S195>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Ideal P Gain/Passthrough'
 * '<S196>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Ideal P Gain Fdbk/Disabled'
 * '<S197>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Integrator/Discrete'
 * '<S198>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Integrator ICs/Internal IC'
 * '<S199>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/N Copy/Disabled'
 * '<S200>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/N Gain/External Parameters'
 * '<S201>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/P Copy/Disabled'
 * '<S202>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Parallel P Gain/External Parameters'
 * '<S203>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Reset Signal/External Reset'
 * '<S204>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Saturation/Passthrough'
 * '<S205>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Saturation Fdbk/Disabled'
 * '<S206>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Sum/Sum_PID'
 * '<S207>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Sum Fdbk/Disabled'
 * '<S208>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tracking Mode/Disabled'
 * '<S209>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tracking Mode Sum/Passthrough'
 * '<S210>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tsamp - Integral/Passthrough'
 * '<S211>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/Tsamp - Ngain/Passthrough'
 * '<S212>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/postSat Signal/Forward_Path'
 * '<S213>' : 'standMode/controller2/initial_state/Drive&Steering
 * torCtl/drive/Discrete PID Controller1/preSat Signal/Forward_Path'
 */
#endif /* RTW_HEADER_standMode_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
