import { cameraRouter } from "~/server/api/routers/camera";
import { correlatedPointsRouter } from "~/server/api/routers/correlatedpoint";
import { visionRouter } from "~/server/api/routers/vision";
import { createTRPCRouter } from "~/server/api/trpc";

/**
 * This is the primary router for your server.
 *
 * All routers added in /api/routers should be manually added here.
 */
export const appRouter = createTRPCRouter({
  camera: cameraRouter,
  correlatedPoints: correlatedPointsRouter,
  vision: visionRouter,
});

// export type definition of API
export type AppRouter = typeof appRouter;
