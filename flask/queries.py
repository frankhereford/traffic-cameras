import asyncio
from prisma import Prisma
import logging

logging.basicConfig(level=logging.INFO)


async def getOrCreateCameraById(db, camera_id: int) -> None:
    await db.connect()
    camera = await db.camera.find_first(where={"coaId": camera_id})

    if camera is None:
        camera = await db.camera.create({"coaId": camera_id})

    await db.disconnect()
    return camera


async def getOrCreateStatusByName(db, camera_id: int, name: str) -> None:
    await db.connect()
    status = await db.status.find_first(where={"name": name})
    camera = await db.camera.find_first(where={"coaId": camera_id})

    if status is None:
        status = await db.status.create({"name": name})

    logging.info("status: %s", status.id)

    await db.camera.update(
        where={"id": camera.id},
        data={"statusId": status.id},
    )

    await db.disconnect()
    return status
