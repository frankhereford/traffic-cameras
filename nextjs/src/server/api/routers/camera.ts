import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const cameraRouter = createTRPCRouter({
  getCameras: protectedProcedure
    .input(z.object({ cameras: z.array(z.number()) }))
    .query(async ({ ctx, input }) => {
      const cameras = await ctx.db.camera.findMany({
        where: {
          coaId: {
            in: input.cameras,
          },
        },
        include: {
          status: true,
        },
      })
      return cameras
    }),
})
