import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const cameraRouter = createTRPCRouter({
  // should be get getSpecificCameras by array of ids
  getCameras: publicProcedure
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
  getAllCameras: publicProcedure
    .input(z.object({}))
    .query(async ({ ctx }) => {
      // Exclude cameras whose most recent status is 'unavailable' and that status is more than 24 hours old
      const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000)

      const cameras = await ctx.db.camera.findMany({
        where: {
          OR: [
            // Cameras whose status is not 'unavailable'
            {
              status: {
                name: {
                  not: "unavailable",
                },
              },
            },
            // Cameras whose status is 'unavailable' but updated within the last 24 hours
            {
              status: {
                name: "unavailable",
              },
              updatedAt: {
                gte: twentyFourHoursAgo,
              },
            },
          ],
        },
        include: {
          status: true,
        },
      })
      return cameras
    }),
  getWorkingCameras: publicProcedure
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
          _count: {
            select: { Location: true },
          },
        },
      })
      return cameras
    }),
})
