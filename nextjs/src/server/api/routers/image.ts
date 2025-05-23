import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const imageRouter = createTRPCRouter({
  getDetections: publicProcedure
    .input(z.object({ camera: z.number() }))
    .query(async ({ ctx, input }) => {
      const camera = await ctx.db.camera.findUnique({
        where: {
          coaId: input.camera,
        },
      })
      // console.log("\ncamera", camera)
      const image = await ctx.db.image.findFirst({
        where: {
          cameraId: camera?.id,
          detections: {
            every: {
              createdAt: {
                gte: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
              },
            },
          },
        },
        include: {
          detections: true,
        },
        orderBy: {
          createdAt: "desc",
        },
      })
      // console.log("image", image)
      return image
    }),
})
