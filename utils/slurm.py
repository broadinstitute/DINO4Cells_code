from logging import getLogger
import os
import signal
import time


logger = getLogger()


def trigger_job_requeue():
    """ Submit a new job to resume from checkpoint.
        Be careful to use only for main process.
    """
    if (
        int(os.environ["SLURM_PROCID"]) == 0
        and str(os.getpid()) == os.environ["MAIN_PID"]
    ):
        print("time is up, back to slurm queue", flush=True)
        command = "scontrol requeue " + os.environ["SLURM_JOB_ID"]
        print(command)
        if os.system(command):
            raise RuntimeError("requeue failed")
        print("New job submitted to the queue", flush=True)
    exit(0)


def SIGTERMHandler(a, b):
    print("received sigterm")
    pass


def signalHandler(a, b):
    print("Signal received", a, time.time(), flush=True)
    os.environ["SIGNAL_RECEIVED"] = "True"
    return


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    os.environ["SIGNAL_RECEIVED"] = "False"
    os.environ["MAIN_PID"] = str(os.getpid())

    signal.signal(signal.SIGUSR1, signalHandler)
    signal.signal(signal.SIGTERM, SIGTERMHandler)
    print("Signal handler installed.", flush=True)