#!/usr/bin/python

from datetime import datetime

from prefect import flow, get_run_logger
from prefect.deployments import Deployment
from dateutil.relativedelta import relativedelta
from prefect.orion.schemas.schedules import CronSchedule

from train import study_and_train
from get_data import get_data


@flow(name='scheduled_training')
def scheduler(date=None):
    logger = get_run_logger()
    if date is None:
        day = datetime.today()
        date = day.strftime("%Y-%m-%d")

    data_date = day - relativedelta(months=1)
    logger.info(f"Getting data for month: {str(data_date.month).zfill(2)}")
    get_data()
    logger.info(f"Training on: {date}")
    study_and_train()


deployment = Deployment.build_from_flow(
    flow=scheduler,
    name="model_training",
    schedule=CronSchedule(cron="0 0 * * 1"),
    work_queue_name="main",
)

if __name__ == '__main__':
    deployment.apply()
