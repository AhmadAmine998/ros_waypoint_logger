import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
)
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from waypoints_loader import load_waypoints


def spline_sample_closed(control_pts, num_points, cs_vel, s_max):
    """
    Sample a closed-loop periodic spline for position (x,y)
    and velocity v as a joint multivariate periodic spline.
    Returns arrays xs, ys, vs, and cumulative arc-length S.
    """
    # Prepare input points and ensure closure
    pts = np.array(control_pts)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    # Compute chord lengths and parameter t
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    dist = np.hypot(dx, dy)
    t = np.concatenate(([0], np.cumsum(dist)))
    total = t[-1]
    if total == 0:
        # Degenerate: all points identical
        xs = np.full(num_points, pts[0, 0])
        ys = np.full(num_points, pts[0, 1])
        vs = np.full(num_points, cs_vel(0))
    else:
        # Normalize parameter to [0,1]
        t_norm = t / total
        # Compute control-point velocities by mapping to original s domain
        s_ctrl = t_norm * s_max
        v_ctrl = cs_vel(s_ctrl)
        # Ensure velocity periodicity
        v_ctrl[-1] = v_ctrl[0]
        # Build joint periodic spline [x, y, v]
        y_ctrl = np.column_stack((pts[:, 0], pts[:, 1], v_ctrl))
        cs_all = CubicSpline(t_norm, y_ctrl, bc_type="periodic", axis=0)
        # Sample spline uniformly
        ts = np.linspace(0, 1, num_points, endpoint=False)
        samples = cs_all(ts)
        xs = samples[:, 0]
        ys = samples[:, 1]
        vs = samples[:, 2]
    # Close loop by appending first sample
    xs = np.concatenate([xs, xs[:1]])
    ys = np.concatenate([ys, ys[:1]])
    vs = np.concatenate([vs, vs[:1]])
    # Compute cumulative arc-length
    ds = np.hypot(np.diff(xs), np.diff(ys))
    S = np.concatenate(([0], np.cumsum(ds)))
    return xs, ys, vs, S


def insert_nearest(control_pts, new_pt):
    pts = np.array(control_pts)
    x0, y0 = new_pt
    min_dist = float("inf")
    insert_idx = 0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            proj = np.array([x1, y1])
        else:
            t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
            t = np.clip(t, 0, 1)
            proj = np.array([x1 + t * dx, y1 + t * dy])
        d = np.hypot(x0 - proj[0], y0 - proj[1])
        if d < min_dist:
            min_dist = d
            insert_idx = i + 1
    new_ctrl = control_pts.copy()
    new_ctrl.insert(insert_idx, new_pt)
    return new_ctrl


class DraggableScatter(pg.ScatterPlotItem):
    def __init__(self, positions, update_callback):
        super().__init__(
            pos=positions,
            data=list(range(len(positions))),
            pen=pg.mkPen("b"),
            brush=pg.mkBrush("b"),
            size=12,
        )
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.positions = list(positions)
        self.update_callback = update_callback
        self.dragIndex = None

    def mousePressEvent(self, ev):
        view_pt = self.getViewBox().mapSceneToView(ev.scenePos())
        pts = self.pointsAt(view_pt)
        if len(pts) > 0 and ev.button() == Qt.MouseButton.LeftButton:
            self.dragIndex = pts[0].data()
            ev.accept()
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self.dragIndex is None or not (ev.buttons() & Qt.MouseButton.LeftButton):
            ev.ignore()
            return
        view_pt = self.getViewBox().mapSceneToView(ev.scenePos())
        self.positions[self.dragIndex] = (view_pt.x(), view_pt.y())
        self.setData(pos=self.positions, data=list(range(len(self.positions))))
        self.update_callback(self.positions)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self.dragIndex = None
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        view_pt = self.getViewBox().mapSceneToView(ev.scenePos())
        pts = self.pointsAt(view_pt)
        if len(pts) > 0:
            idx = pts[0].data()
            self.positions.pop(idx)
        elif ev.button() == Qt.MouseButton.LeftButton:
            new_pt = (view_pt.x(), view_pt.y())
            self.positions = insert_nearest(self.positions, new_pt)
        else:
            ev.ignore()
            return
        self.setData(pos=self.positions, data=list(range(len(self.positions))))
        self.update_callback(self.positions)
        ev.accept()


class MainWindow(QMainWindow):
    def __init__(self, csv_path="../traj_examples/new_loc.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.setWindowTitle("Race Line Editor: Circuit Mode")
        self.num_samples = 30

        df = load_waypoints(self.csv_path)
        self.static_pts = list(df[["x_m", "y_m"]].itertuples(index=False, name=None))
        if not self.static_pts:
            self.static_pts = [(0, 0), (1, 1), (2, 0), (3, 1)]

        s_orig = df["s_m"].to_numpy()
        v_orig = df["vx_mps"].to_numpy()
        self.v_orig = v_orig  # store original velocity profile
        self.cs_vel = CubicSpline(s_orig, v_orig)
        self.s_max = s_orig[-1]

        container = QWidget()
        main_layout = QVBoxLayout(container)

        ctrl_layout = QHBoxLayout()
        save_btn = QPushButton("Save Waypoints")
        save_btn.clicked.connect(self.save_csv)
        ctrl_layout.addWidget(save_btn)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

        plot_layout = QHBoxLayout()
        self.plot = pg.PlotWidget()
        vb = self.plot.getViewBox()
        vb.setAspectLocked(True)
        vb.setMouseMode(pg.ViewBox.RectMode)
        plot_layout.addWidget(self.plot, stretch=1)
        slider_layout = QVBoxLayout()
        self.slider_label = QLabel(f"Points: {self.num_samples}")
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(1, 50)
        self.slider.setValue(self.num_samples)
        self.slider.valueChanged.connect(self.on_slider_change)
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addStretch()
        plot_layout.addLayout(slider_layout)
        main_layout.addLayout(plot_layout, stretch=3)

        self.vel_plot = pg.PlotWidget()
        self.vel_plot.setTitle("Velocity Profile")
        self.vel_plot.setLabel("bottom", "s (m)")
        self.vel_plot.setLabel("left", "vₓ (m/s)")
        self.vel_plot.plot(s_orig, v_orig, pen=pg.mkPen("lightgray", width=2))
        self.vel_spline = pg.PlotDataItem(pen=pg.mkPen("r", width=4))
        self.vel_plot.addItem(self.vel_spline)
        self.vel_ctrl_scatter = pg.ScatterPlotItem(
            symbol="o", pen=pg.mkPen("r"), brush=pg.mkBrush("r"), size=12
        )
        self.vel_plot.addItem(self.vel_ctrl_scatter)
        self.vel_plot.addItem(
            pg.InfiniteLine(
                pos=0, angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine)
            )
        )
        main_layout.addWidget(self.vel_plot, stretch=3)

        static_scatter = pg.ScatterPlotItem(
            pos=self.static_pts,
            pen=pg.mkPen("lightgray"),
            brush=pg.mkBrush("lightgray"),
            size=8,
        )
        self.plot.addItem(static_scatter)

        xs, ys, vs, S = spline_sample_closed(
            self.static_pts, self.num_samples, self.cs_vel, self.s_max
        )
        self.ctrl_pts = list(zip(xs[:-1], ys[:-1]))
        self.draggable = DraggableScatter(self.ctrl_pts, self.update_spline)
        self.plot.addItem(self.draggable)

        self.spline_curve = pg.PlotDataItem(pen=pg.mkPen("r", width=4))
        self.plot.addItem(self.spline_curve)
        self.update_spline(self.ctrl_pts)

        self.setCentralWidget(container)

    def on_slider_change(self, value):
        self.num_samples = value
        xs, ys, vs, S = spline_sample_closed(
            self.static_pts, self.num_samples, self.cs_vel, self.s_max
        )
        self.ctrl_pts = list(zip(xs[:-1], ys[:-1]))
        self.draggable.positions = self.ctrl_pts.copy()
        self.draggable.setData(pos=self.ctrl_pts, data=list(range(len(self.ctrl_pts))))
        self.update_spline(self.ctrl_pts)

    def update_spline(self, ctrl_pts):
        xs, ys, vs, S = spline_sample_closed(
            ctrl_pts, max(200, len(ctrl_pts) * 10), self.cs_vel, self.s_max
        )
        self.spline_curve.setData(xs, ys)
        self.vel_spline.setData(S, vs)
        pts_ctrl = np.array(ctrl_pts)
        if len(pts_ctrl) > 1:
            d_ctrl = np.hypot(np.diff(pts_ctrl[:, 0]), np.diff(pts_ctrl[:, 1]))
            S_ctrl = np.concatenate(([0], np.cumsum(d_ctrl)))
        else:
            S_ctrl = np.array([0])
        static_arr = np.array(self.static_pts)
        # compute distances between ctrl_pts and static waypoints
        dists = np.linalg.norm(static_arr[None, :, :] - pts_ctrl[:, None, :], axis=2)
        idxs = np.argmin(dists, axis=1)
        V_ctrl = self.v_orig[idxs]
        self.vel_ctrl_scatter.setData(S_ctrl, V_ctrl)
        count = len(ctrl_pts)
        self.slider.blockSignals(True)
        self.slider.setValue(min(count, self.slider.maximum()))
        self.slider.blockSignals(False)
        self.slider_label.setText(f"Points: {count}")

    def save_csv(self):
        # Generate a new filename by appending '_modified' to the original filename
        new_csv_path = self.csv_path.replace('.csv', '_modified.csv')

        # sample the current spline before saving
        xs, ys, vs, S = spline_sample_closed(
            self.ctrl_pts, self.num_samples, self.cs_vel, self.s_max
        )
        # compute heading psi
        x_np = xs
        y_np = ys
        vx_np = vs
        dx = np.gradient(x_np, edge_order=2)
        dy = np.gradient(y_np, edge_order=2)
        psi = np.arctan2(dy, dx)
        # compute curvature kappa
        x_ext = np.concatenate((x_np[-2:], x_np, x_np[:2]))
        y_ext = np.concatenate((y_np[-2:], y_np, y_np[:2]))
        dx_dt = np.gradient(x_ext, edge_order=2)
        dy_dt = np.gradient(y_ext, edge_order=2)
        d2x = np.gradient(dx_dt, edge_order=2)
        d2y = np.gradient(dy_dt, edge_order=2)
        denom = (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        denom[denom == 0] = np.finfo(float).eps
        kappa_full = (dx_dt * d2y - d2x * dy_dt) / denom
        kappa = kappa_full[2:-2]
        # compute acceleration ax
        ax = np.gradient(vx_np, edge_order=2)
        # prepare DataFrame
        df_out = pd.DataFrame(
            {
                "s_m": S,
                "x_m": x_np,
                "y_m": y_np,
                "psi_rad": psi,
                "kappa_radpm": kappa,
                "vx_mps": vx_np,
                "ax_mps2": ax,
            }
        )
        # write comment header lines
        with open(new_csv_path, "w") as f:
            f.write("#\n")
            f.write("#\n")
            f.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        # append data rows in CSV format
        df_out.to_csv(
            new_csv_path,
            sep=";",
            index=False,
            header=False,
            mode="a",
            float_format="%.6f",
        )
        print(f"Saved {len(df_out)} interpolated points to {new_csv_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec())
