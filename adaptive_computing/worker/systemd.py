"""
systemd.py — Generate systemd unit files for HeroWorker daemons.

Usage::

    from adaptive_computing.worker.systemd import generate_unit

    unit = generate_unit(
        description="Gates OpenStack worker on Gila",
        user="gates",
        work_dir="/opt/gates-assistant",
        python="/opt/gates-assistant/.venv/bin/python",
        module="gates_worker",
    )
    print(unit)
    # Write to /etc/systemd/system/gates-worker.service, then:
    # sudo systemctl daemon-reload && sudo systemctl enable --now gates-worker
"""

from __future__ import annotations

_UNIT_TEMPLATE = """\
[Unit]
Description={description}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={work_dir}
ExecStart={python} -m {module}{args_str}
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier={syslog_id}

[Install]
WantedBy=multi-user.target
"""


def generate_unit(
    description: str,
    user: str,
    work_dir: str,
    python: str,
    module: str,
    args: str = "",
    syslog_id: str = "",
) -> str:
    """Return a systemd unit file for a HeroWorker daemon.

    Args:
        description: Human-readable unit description (shown in ``systemctl status``).
        user:        System user to run the service as.
        work_dir:    Absolute working directory for the process.
        python:      Absolute path to the Python interpreter (e.g. venv python).
        module:      Python module to invoke with ``python -m <module>``.
        args:        Extra CLI arguments appended to ExecStart.
        syslog_id:   ``SyslogIdentifier`` tag in the journal (defaults to *module*).

    Returns:
        Complete unit file content as a string.

    Example::

        unit = generate_unit(
            description="Gates OpenStack worker",
            user="gates",
            work_dir="/opt/gates-assistant",
            python="/opt/gates-assistant/.venv/bin/python",
            module="gates_worker",
            args="--log-level DEBUG",
        )
        Path("/etc/systemd/system/gates-worker.service").write_text(unit)
    """
    return _UNIT_TEMPLATE.format(
        description=description,
        user=user,
        work_dir=work_dir,
        python=python,
        module=module,
        args_str=f" {args}" if args else "",
        syslog_id=syslog_id or module,
    )
