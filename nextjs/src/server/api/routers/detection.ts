import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const detectionRouter = createTRPCRouter({
  getDetections: protectedProcedure
    .input(z.object({ camera: z.number() }))
    .query(async ({ ctx, input }) => {
      const camera = await ctx.db.camera.findFirstOrThrow({
        where: { coaId: input.camera },
      })
      const image = await ctx.db.image.findFirst({
        where: { cameraId: camera.id },
        include: { detections: true },
        orderBy: { createdAt: "desc" },
      })
      return image
    }),
  getHistoricCameraDetections: protectedProcedure
    .input(z.object({ camera: z.number() }))
    .query(async ({ ctx, input }) => {
      const camera = await ctx.db.camera.findFirstOrThrow({
        where: { coaId: input.camera },
      })
      const images = await ctx.db.image.findMany({
        where: { cameraId: camera.id },
        include: { detections: true },
        orderBy: { createdAt: "desc" },
      })

      const validLabels = [
        "car",
        "person",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
      ]

      let totalDetections = 0
      const limitedImages = images.reduce((acc: typeof images, image) => {
        const validDetections = image.detections.filter(
          (detection) =>
            validLabels.includes(detection.label) &&
            (true || detection.isInsideConvexHull), // set to false to only have inside the convex hull
        )

        if (totalDetections + validDetections.length > 250) {
          return acc
        }

        totalDetections += validDetections.length
        acc.push({
          ...image,
          detections: validDetections.map((detection) => {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const { picture, ...detectionWithoutPicture } = detection
            return { ...detectionWithoutPicture, picture: null }
          }),
        })

        return acc
      }, [])

      return limitedImages
    }),
})
