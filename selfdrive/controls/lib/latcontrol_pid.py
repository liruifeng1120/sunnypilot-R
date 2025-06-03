import math  
import numpy as np  
  
from cereal import log  
from openpilot.selfdrive.controls.lib.latcontrol import LatControl  
from openpilot.common.pid import PIDController  
  
  
class LatControlPID(LatControl):  
  def __init__(self, CP, CP_SP, CI):  
    super().__init__(CP, CP_SP, CI)  
      
    # ====== 针对高速大弯优化的补偿参数 ======  
    # 针对120km/h高速场景进一步优化  
    kpV_comp = [-0.1, -0.2, -0.4, -0.7]  # 进一步降低高速比例增益，减少过度转向  
    kiV_comp = [-0.01, -0.02, -0.03, -0.05]  # 加大积分增益补偿，抑制震荡  
    kf_comp = -0.15  # 进一步降低前馈增益，减少高速过冲  
  
    # ====== 应用补偿 ======  
    new_kpV = [x + y for x, y in zip(CP.lateralTuning.pid.kpV, kpV_comp)]  
    new_kiV = [x + y for x, y in zip(CP.lateralTuning.pid.kiV, kiV_comp)]  
    new_kf = CP.lateralTuning.pid.kf + kf_comp  
  
    self.pid = PIDController(  
        (CP.lateralTuning.pid.kpBP, new_kpV),  # 应用补偿后的比例增益  
        (CP.lateralTuning.pid.kiBP, new_kiV),  # 应用补偿后的积分增益  
        k_f=new_kf,  # 应用补偿后的前馈增益  
        pos_limit=self.steer_max,  
        neg_limit=-self.steer_max  
    )  
    # =====================  
  
    self.get_steer_feedforward = CI.get_steer_feedforward_function()  
  
  def get_speed_compensation(self, speed_ms):  
    """基于实时速度的动态补偿"""  
    if speed_ms > 30:  # 约108km/h以上  
        return -0.2  # 额外降低响应  
    elif speed_ms > 25:  # 约90km/h以上  
        return -0.1  
    return 0.0  
  
  def reset(self):  
    super().reset()  
    self.pid.reset()  
  
  def update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited):  
    pid_log = log.ControlsState.LateralPIDState.new_message()  
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)  
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)  
  
    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))  
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg  
    error = angle_steers_des - CS.steeringAngleDeg  
  
    # 添加曲率敏感补偿（针对大弯）  
    curvature_compensation = 0.0  
    if abs(desired_curvature) > 0.05:  # 大弯判断  
        curvature_compensation = -0.1 * abs(desired_curvature)  
      
    # 应用速度和曲率补偿  
    speed_compensation = self.get_speed_compensation(CS.vEgo)  
    total_compensation = speed_compensation + curvature_compensation  
    error = error * (1.0 + total_compensation)  
  
    pid_log.steeringAngleDesiredDeg = angle_steers_des  
    pid_log.angleError = error  
    if not active:  
      output_steer = 0.0  
      pid_log.active = False  
      self.pid.reset()  
    else:  
      steer_feedforward = self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)  
  
      output_steer = self.pid.update(error, override=CS.steeringPressed,  
                                     feedforward=steer_feedforward, speed=CS.vEgo)  
  
      # === 修正后的速度敏感型积分限幅 ===  
      if CS.vEgo > 30.0:  # 120km/h对应约33m/s，提前限制  
          max_i = 0.05 * abs(output_steer)  # 进一步限制积分累积  
          self.pid.i = np.clip(self.pid.i, -max_i, max_i)  
      elif CS.vEgo > 25.0:  
          max_i = 0.1 * abs(output_steer)  
          self.pid.i = np.clip(self.pid.i, -max_i, max_i)  
      elif CS.vEgo > 15.0:  
          max_i = 0.2 * abs(output_steer)  
          self.pid.i = np.clip(self.pid.i, -max_i, max_i)  
      # =========================  
  
      pid_log.active = True  
      pid_log.p = float(self.pid.p)  
      pid_log.i = float(self.pid.i)  
      pid_log.f = float(self.pid.f)  
      pid_log.output = float(output_steer)  
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS, steer_limited_by_controls, curvature_limited))  
  
    return output_steer, angle_steers_des, pid_log
