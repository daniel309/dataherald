from modules.db_connection.models.entities import (
    DBConnection,
    Driver,
)


class DBConnectionResponse(DBConnection):
    metadata: dict | None

    def dict(self, **kwargs):
        dic = super().dict(**kwargs)
        if "metadata" in dic and dic["metadata"] and "dh_internal" in dic["metadata"]:
            del dic["metadata"]["dh_internal"]
        return dic


class DriverResponse(Driver):
    pass
