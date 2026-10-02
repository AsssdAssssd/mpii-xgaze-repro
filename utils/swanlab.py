import swanlab
from swanlab.plugin.notification import LarkCallback

WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/d00b7a97-2138-4105-8330-864d63f4d536"
SECRET = "xgChT9EMy1ZJFDiKB5VeSc"


class _MessageLark(LarkCallback):
    def __init__(self, message):
        super().__init__(webhook_url=WEBHOOK_URL, secret=SECRET)
        self._message = message

    def _build_content(self, state, error):
        return f"state :{state}  \n  error :{error}  \n info: {self._message}"


def notify(message, project="mpii_xgaze"):
    swanlab.init(
        mode="disabled",
        project=project,
        name="notify",
        callbacks=[_MessageLark(str(message))],
    )
    swanlab.finish()
