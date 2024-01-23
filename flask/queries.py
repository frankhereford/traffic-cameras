from prisma import Prisma
import logging

db = Prisma(auto_register=True)

db.connect()

logging.basicConfig(level=logging.INFO)


async def getOrCreateCameraById(camera_id: int) -> None:
    # db.connect()
    camera = db.camera.find_first(where={"coaId": camera_id})

    if camera is None:
        camera = db.camera.create({"coaId": camera_id})

    # db.disconnect()
    return camera


async def getOrCreateStatusByName(camera_id: int, name: str) -> None:
    # db.connect()
    status = db.status.find_first(where={"name": name})
    camera = db.camera.find_first(where={"coaId": camera_id})

    if status is None:
        status = db.status.create({"name": name})

    logging.info("status: %s", status.id)

    db.camera.update(
        where={"id": camera.id},
        data={"statusId": status.id},
    )

    # db.disconnect()
    return status
