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
      console.log("input", input);

      const camera = await ctx.db.camera.findFirst({
        where: { cameraId: input.cameraId },
      });

      if (!camera) {
        throw new Error("Camera not found");
      }

      console.log("camera", camera);

      const newPoint = await ctx.db.correlatedPoint.create({
        data: {
          cameraX: input.cameraX,
          cameraY: input.cameraY,
          mapLatitude: input.mapLat,
          mapLongitude: input.mapLng,
          cameraId: camera.id,
        },
      });

      console.log("newpoint", newPoint);
    }),
});
