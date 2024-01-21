import { z } from "zod";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";

export const correlatedPointsRouter = createTRPCRouter({
  setPointPair: protectedProcedure
    .input(
      z.object({
        cameraId: z.number(),
        cameraX: z.number(),
        cameraY: z.number(),
        mapLat: z.number(),
        mapLng: z.number(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const camera = await ctx.db.camera.findFirst({
        where: { cameraId: input.cameraId },
      });

      if (!camera) {
        throw new Error("Camera not found");
      }

      const newPoint = await ctx.db.correlatedPoint.create({
        data: {
          cameraX: input.cameraX,
          cameraY: input.cameraY,
          mapLatitude: input.mapLat,
          mapLongitude: input.mapLng,
          cameraId: camera.id,
        },
      });
    }),

  getPointPairs: protectedProcedure
    .input(z.object({ cameraId: z.number() }))
    .query(async ({ ctx, input }) => {
      const camera = await ctx.db.camera.findFirst({
        where: { cameraId: input.cameraId },
      });

      if (!camera) {
        throw new Error("Camera not found");
      }

      const pointPairs = await ctx.db.correlatedPoint.findMany({
        where: { cameraId: camera.id },
      });

      return pointPairs;
    }),
});
