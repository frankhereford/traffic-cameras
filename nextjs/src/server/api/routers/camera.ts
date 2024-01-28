import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const cameraRouter = createTRPCRouter({
  // should be get getSpecificCameras by array of ids
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
  getAllCameras: protectedProcedure
    .input(z.object({}))
    .query(async ({ ctx }) => {
      const cameras = await ctx.db.camera.findMany({})
      return cameras
    }),
  getWorkingCameras: protectedProcedure
    .input(z.object({}))
    .query(async ({ ctx, input }) => {
      const cameras = await ctx.db.camera.findMany({
        where: {
          status: {
            name: "ok",
          },
        },
        include: {
          status: true,
        },
      })
      return cameras
    }),
})
