import math

from cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController


class LatControlPID(LatControl):
  def __init__(self, CP, CP_SP, CI):
    super().__init__(CP, CP_SP, CI)
    
    # ====== 新增补偿参数 ======
    # 建议调整范围：±0.1~0.5（首次调试建议按注释值）
    kpV_comp = [-0.1, -0.1, -0.3, -0.5]  # 比例增益补偿（按车速段分段补偿，最后两位对应高速段）
    kiV_comp = [-0.01, -0.02, -0.02, -0.03]  # 积分增益补偿（抑制低速震荡）
    kf_comp = -0.1  # 前馈增益补偿（减少高速过冲）

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

    pid_log.steeringAngleDesiredDeg = angle_steers_des
    pid_log.angleError = error
    if not active:
      output_steer = 0.0
      pid_log.active = False
      self.pid.reset()
    else:
      steer_feedforward = self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)

      # === 新增速度敏感型积分限幅 ===
      if CS.vEgo > 25.0:  # 高速时限制积分累积
        self.pid.integrator_max = 0.1 * abs(output_steer)
      elif CS.vEgo > 15.0:
        self.pid.integrator_max = 0.2 * abs(output_steer)
      # =========================

      output_steer = self.pid.update(error, override=CS.steeringPressed,
                                     feedforward=steer_feedforward, speed=CS.vEgo)
      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(output_steer)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS, steer_limited_by_controls, curvature_limited))

    return output_steer, angle_steers_des, pid_log
