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
      return true
      // const cameras = await ctx.db.camera.findMany({
      //   where: {
      //     coaId: {
      //       in: input.cameras,
      //     },
      //   },
      //   include: {
      //     status: true,
      //   },
      // })
      // return cameras
    }),
})
