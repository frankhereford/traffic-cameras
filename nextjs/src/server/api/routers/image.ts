import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const imageRouter = createTRPCRouter({
  getDetections: protectedProcedure
    .input(z.object({ camera: z.number() }))
    .query(async ({ ctx, input }) => {
      const camera = await ctx.db.camera.findUnique({
        where: {
          coaId: input.camera,
        },
      })
      console.log("\ncamera", camera)
      const image = await ctx.db.image.findFirst({
        where: {
          cameraId: camera?.id,
        },
        include: {
          detections: true,
        },
        orderBy: {
          createdAt: "desc",
        },
      })
      console.log("image", image)
      return image
    }),
})
