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
})
